---
title: Incremental Decoding
date: 2020-03-17 15:27:27
categories: NLP
tags:
  - inference
  - math
  - seq2seq
mathjax: true
---

Record the incremental decoding processing of parallel decoding models such as CNN seq2seq and Transformer in the inference phase in Fairseq.

<!--more-->

{% language_switch %}

{% lang_content en %}
# Fairseq Architecture

- In Facebook's seq2seq library Fairseq, all models inherit the FairseqEncoderDecoder class, all Encoders inherit the FairseqEncoder class, all Decoders inherit the FairseqIncrementalDecoder class, and FairseqIncrementalDecoder inherits from the FairseqDecoder class.
- The FairseqEncoder class only defines forward, reorder_encoder_out, max_positions, and upgrade_state_dict, with forward being the most important method, defining the forward propagation process of encoding. Reorder is actually more important in the decoder, but it is defined here as well.
- The FairseqDecoder class defines forward, extract_features, output_layer, get_normalized_probs, max_positions, upgrade_state_dict, and prepare_for_onnx_export_. forward = extract_features + output_layer, which means forward defines the entire forward process of decoding a sequence, while extract_features only defines obtaining the decoder's state sequence.
- The Incremental Decoder additionally defines reorder_incremental_state and set_beam_size. Reorder is closely related to incremental decoding and beam search, which will be detailed later.

# Training Parallelization and Inference Incrementation

- Models like CNN seq2seq and Transformer break the sequentiality of RNN models, enabling the encoder and decoder in the seq2seq architecture to be trained in parallel during training.
- Parallel training of the encoder is quite obvious, and the decoder is essentially a language model that can be parallelized during training because of teacher forcing, where the input at each time step is assumed to be known. Thus, the entire decoder input of (Batch, Length, Hidden) can be directly input into the model for training.
- However, during testing (inference), the input at each time step is determined by the output of the previous time step, which cannot be parallelized. If the entire decoder is run repeatedly, it would run Length times, and only the information of the first i positions is useful in the i-th run, with the remaining calculations completely wasted, significantly reducing inference efficiency.
- At this point, incremental decoding becomes necessary. During the inference phase, whether it's CNN or Transformer, decoding is done step by step like an RNN, using information previously inferred at each step, rather than starting from scratch.

# CNN

- For CNN, it can be observed that at each layer of the decoder, the i-th position only needs information from the [i-k, i) positions, where k is the window size of the one-dimensional convolution. Therefore, by maintaining a queue of length k to save the states calculated by each layer, the model can reuse information previously inferred.

- Each calculation only needs to decode the i-th position, i.e., operate on (Batch, 1, Hidden) data Length times.

- In the code, FConvDecoder passes input x and incremental_state to LinearizedConvolution, which is described as:
  
  ```
  """An optimized version of nn.Conv1d.
  At training time, this module uses ConvTBC, which is an optimized version
  of Conv1d. At inference time, it optimizes incremental generation (i.e.,
  one time step at a time) by replacing the convolutions with linear layers.
  Note that the input order changes from training to inference.
  """
  ```

- During training, data is organized in a Time-First format for convolution to fully utilize GPU parallel performance. During inference, convolution layers are replaced with equivalent linear layers for frame-by-frame inference
  
  ```
  if incremental_state is None:
     output = super().forward(input) # Here, LinearizedConvolution's parent class is ConvTBC, so if there's no inference, the entire sequence is sent to ConvTBC
     if self.kernel_size[0] > 1 and self.padding[0] > 0:
         # remove future timesteps added by padding
         output = output[:-self.padding[0], :, :]
  return output
  ```

- Otherwise, inference is done layer by layer using linear layers, and the input buffer is updated to update incremental_state
  
  ```
  # reshape weight
  weight = self._get_linearized_weight()
  kw = self.kernel_size[0]
  bsz = input.size(0)  # input: bsz x len x dim
  if kw > 1:
     input = input.data
     input_buffer = self._get_input_buffer(incremental_state)
     if input_buffer is None:
         input_buffer = input.new(bsz, kw, input.size(2)).zero_()
         self._set_input_buffer(incremental_state, input_buffer)
     else:
         # shift buffer
         input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
     # append next input
     input_buffer[:, -1, :] = input[:, -1, :]
     input = input_buffer
  with torch.no_grad():
     output = F.linear(input.view(bsz, -1), weight, self.bias)
  return output.view(bsz, 1, -1)
  ```

# Transformer

- Similarly, let's look at how self-attention-based models maintain an incremental state

- Clearly, when inferring the token at the i-th position, it's not just related to the history of a window size like CNN, but to the first i-1 positions. However, note that the key and value computed for the first i-1 positions remain unchanged and can be reused. The i-th position only generates its own key, value, and query, and uses the query to query itself and the reusable key and value of the first i-1 positions. Therefore, the incremental state should include key and value information, maintaining not a window size, but the entire sequence.

- In the code, TransformerDecoder passes the current layer input and encoder output to TransformerDecoderLayer, updating the buffer
  
  ```
  if prev_self_attn_state is not None:
     prev_key, prev_value = prev_self_attn_state[:2]
     saved_state: Dict[str, Optional[Tensor]] = {
         "prev_key": prev_key,
         "prev_value": prev_value,
     }
     if len(prev_self_attn_state) >= 3:
         saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
     assert incremental_state is not None
     self.self_attn._set_input_buffer(incremental_state, saved_state)
  _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
  ```

- And in MultiHeadAttention, if incremental_state exists, set key and value to None, and subsequent calculations will skip when they are None
  
  ```
  if incremental_state is not None:
     saved_state = self._get_input_buffer(incremental_state)
     if saved_state is not None and "prev_key" in saved_state:
         # previous time steps are cached - no need to recompute
         # key and value if they are static
         if static_kv:
             assert self.encoder_decoder_attention and not self.self_attention
             key = value = None
  ```

- Then read, calculate, and update, with detailed code
  
  ```
  if saved_state is not None:
     # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
     if "prev_key" in saved_state:
         _prev_key = saved_state["prev_key"]
         assert _prev_key is not None
         prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
         if static_kv:
             k = prev_key
         else:
             assert k is not None
             k = torch.cat([prev_key, k], dim=1)
     if "prev_value" in saved_state:
         _prev_value = saved_state["prev_value"]
         assert _prev_value is not None
         prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
         if static_kv:
             v = prev_value
         else:
             assert v is not None
             v = torch.cat([prev_value, v], dim=1)
     prev_key_padding_mask: Optional[Tensor] = None
     if "prev_key_padding_mask" in saved_state:
         prev_key_padding_mask = saved_state["prev_key_padding_mask"]
     assert k is not None and v is not None
     key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
         key_padding_mask=key_padding_mask,
         prev_key_padding_mask=prev_key_padding_mask,
         batch_size=bsz,
         src_len=k.size(1),
         static_kv=static_kv,
     )
  
     saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
     saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
     saved_state["prev_key_padding_mask"] = key_padding_mask
     # In this branch incremental_state is never None
     assert incremental_state is not None
     incremental_state = self._set_input_buffer(incremental_state, saved_state)
  ```

# Generate

- Fairseq's models define all forward processes, and which forward process is used depends on whether it's training or inference. Inference uses fairseq-generate.

- To complete a seq2seq, you need to specify the task and model, along with other learning hyperparameters. The task determines dataset parameters, establishes evaluation metrics, vocabulary, data batches, and model instantiation.

- The most important parts are train_step and inference_step, let's look at inference_step
  
  ```
  def inference_step(self, generator, models, sample, prefix_tokens=None):
     with torch.no_grad():
         return generator.generate(models, sample, prefix_tokens=prefix_tokens)
  ```

- Here, the generator is a sequence_generator object, with the generation part
  
  ```
  for step in range(max_len + 1):  # one extra step for EOS marker
     # reorder decoder internal states based on the prev choice of beams
     if reorder_state is not None:
         if batch_idxs is not None:
             # update beam indices to take into account removed sentences
             corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
             reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
         model.reorder_incremental_state(reorder_state)
         encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)
  
     lprobs, avg_attn_scores = model.forward_decoder(
         tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
     )
  ```

- This wraps an ensemble model. If we only have one decoder model, then forward_decoder actually executes
  
  ```
  def _decode_one(
     self, tokens, model, encoder_out, incremental_states, log_probs,
     temperature=1.,
  ):
     if self.incremental_states is not None:
         decoder_out = list(model.forward_decoder(
             tokens,
             encoder_out=encoder_out,
             incremental_state=self.incremental_states[model],
         ))
     else:
         decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
  ```

- Here you can see that incremental decoding is used to decode the sentence step by step

# Reorder

- For more detailed information, refer to this blog post, which is very well written and even officially endorsed by being added to the code comments [understanding-incremental-decoding-in-fairseq](http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/)
- There's another point about reorder in the decoder, also mentioned in this blog post.
- During inference, unlike training, beam search is used. So we maintain not just one cache queue, but beam_size number of queues.
- When selecting the i-th word, the input token stored in the k-th beam's cache queue might have come from the j-th beam's cache queue during beam search at the i-1 position. Therefore, reordering is needed to ensure consistency.
{% endlang_content %}

{% lang_content zh %}
# Fairseq架构

- 在Facebook推出的seq2seq库Fairseq当中，所有模型继承FairseqEncdoerDecoder类，所有的Encoder继承FairseqEncoder类，所有的Decoder继承FairseqIncrementalDecoder类，而FairseqIncrementalDecoder继承自FairseqDecoder类。
- FairseqEncoder类只定义了forward，reorder_encoder_out,max_positions，upgrade_state_dict，最重要的就是forward，即定义编码的前向传播过程。reorder其实在decoder中更重要，但是这里也定义了。
- FairseqDecoder类定义了forward，extract_features,output_layer，get_normalized_probs，max_positions，upgrade_state_dict，prepare_for_onnx_export_。forward=extract_features+output_layer，即forward定义了解码出序列的整个前向过程，而extract_features只定义到获得整个decoder的state sequence。
- Incremental Decoder额外定义了reorder_incremental_state，set_beam_size。reorder是和incremental以及beam search密切相关的，后文将详细介绍。

# 训练并行，推理增量

- 像CNN seq2seq， Transformer之类的模型打破了RNN模型的顺序性，使得seq2seq架构中的编码器和解码器在训练是都可以并行训练。
- 编码器并行训练非常显然，而解码器实际上是一个语言模型，之所以可以并行是因为在训练时采用了teacher forcing，因此语言模型的每一时间步输入在训练时我们假设是已知的，就可以一整个(Batch,Length,Hidden)的decoder input输入模型，直接训练。
- 但是在测试（推理）阶段，每一时间步的输入由上一时间步的输出决定，无法并行操作，如果反复运行整个decoder，那么就要运行Length次，且第i次只有前i个位置的信息是有用的，剩下部分的计算完全浪费掉了，推理的效率大大降低。
- 这个时候就需要incremental decoding，即在推理阶段，无论是CNN还是Transformer，都想RNN一样一步一步解码，每一步使用之前推理得到的信息，而不是完全从头开始计算。

# CNN

- 对于CNN，可以发现，decoder无论哪一层，第i个位置都只需要该层[i-k,i)位置上内的信息，其中k为一维卷积的窗长。因此，只需要维护一个长度为k的队列，保存各层计算出来的state，就可以复用模型之前推理得到的信息，之后再把当前的state更新到队列中。

- 每次计算时只需要对第i个位置进行decoding，即操作(Batch,1,Hidden)的数据Length次。

- 在代码里，FConvDecoder将输入x和incremental_state一起传给了LinearizedConvolution，这里的介绍是
  
  ```
  """An optimized version of nn.Conv1d.
  At training time, this module uses ConvTBC, which is an optimized version
  of Conv1d. At inference time, it optimizes incremental generation (i.e.,
  one time step at a time) by replacing the convolutions with linear layers.
  Note that the input order changes from training to inference.
  """
  ```

- 即训练时使用Time-First的形式组织数据进行卷积，充分利用GPU的并行性能，在推断时，将卷积层换成相同效果的线性层，逐帧进行推断
  
  ```
  if incremental_state is None:
     output = super().forward(input) # 这里 LinearizedConvolution的父类是ConvTBC，即没有推断时，直接将整个序列送入ConvTBC
     if self.kernel_size[0] > 1 and self.padding[0] > 0:
         # remove future timesteps added by padding
         output = output[:-self.padding[0], :, :]
  return output
  ```

- 否则，就逐层用线性层推断，并更新input buffer进而更新incremental_state
  
  ```
  # reshape weight
  weight = self._get_linearized_weight()
  kw = self.kernel_size[0]
  bsz = input.size(0)  # input: bsz x len x dim
  if kw > 1:
     input = input.data
     input_buffer = self._get_input_buffer(incremental_state)
     if input_buffer is None:
         input_buffer = input.new(bsz, kw, input.size(2)).zero_()
         self._set_input_buffer(incremental_state, input_buffer)
     else:
         # shift buffer
         input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
     # append next input
     input_buffer[:, -1, :] = input[:, -1, :]
     input = input_buffer
  with torch.no_grad():
     output = F.linear(input.view(bsz, -1), weight, self.bias)
  return output.view(bsz, 1, -1)
  ```

# Transformer

- 同样的，我们看基于自注意力的模型如何去维护一个incremental state

- 显然，在推断第i个位置的token时，不像CNN只与窗口大小的history相关，而是与前i-1个位置相关，但是注意，前i-1个位置计算出来的key和value是不变的，是可以复用的，第i位置只生成该位置的key，value以及query，并用query查询自己以及前i-1个位置复用的key,value，因此，incremental state应该包含了key与value的信息，且维护的不是窗口大小，而是整个序列。

- 在代码里，TransformerDecoder将当前层输入和encoder输出传给TransformerDecoderLayer，更新buffer
  
  ```
  if prev_self_attn_state is not None:
     prev_key, prev_value = prev_self_attn_state[:2]
     saved_state: Dict[str, Optional[Tensor]] = {
         "prev_key": prev_key,
         "prev_value": prev_value,
     }
     if len(prev_self_attn_state) >= 3:
         saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
     assert incremental_state is not None
     self.self_attn._set_input_buffer(incremental_state, saved_state)
  _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
  ```

- 并在MultiHeadAttention里，假如incremental_state存在，将key和value设为None，后面的计算判断为None时就跳过计算
  
  ```
  if incremental_state is not None:
     saved_state = self._get_input_buffer(incremental_state)
     if saved_state is not None and "prev_key" in saved_state:
         # previous time steps are cached - no need to recompute
         # key and value if they are static
         if static_kv:
             assert self.encoder_decoder_attention and not self.self_attention
             key = value = None
  ```

- 之后读取、计算、更新，代码写的很详细。
  
  ```
  if saved_state is not None:
     # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
     if "prev_key" in saved_state:
         _prev_key = saved_state["prev_key"]
         assert _prev_key is not None
         prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
         if static_kv:
             k = prev_key
         else:
             assert k is not None
             k = torch.cat([prev_key, k], dim=1)
     if "prev_value" in saved_state:
         _prev_value = saved_state["prev_value"]
         assert _prev_value is not None
         prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
         if static_kv:
             v = prev_value
         else:
             assert v is not None
             v = torch.cat([prev_value, v], dim=1)
     prev_key_padding_mask: Optional[Tensor] = None
     if "prev_key_padding_mask" in saved_state:
         prev_key_padding_mask = saved_state["prev_key_padding_mask"]
     assert k is not None and v is not None
     key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
         key_padding_mask=key_padding_mask,
         prev_key_padding_mask=prev_key_padding_mask,
         batch_size=bsz,
         src_len=k.size(1),
         static_kv=static_kv,
     )
  
     saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
     saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
     saved_state["prev_key_padding_mask"] = key_padding_mask
     # In this branch incremental_state is never None
     assert incremental_state is not None
     incremental_state = self._set_input_buffer(incremental_state, saved_state)
  ```

# Generate

- Fairseq的模型定义了所有前向过程，至于具体选择哪个前向过程则依据训练还是推断来决定。推断使用了fairseq-generate。

- 要完成一次seq2seq，需要指定task和model，以及其他学习超参数。其中task确定了数据集参数，建立评价指标、词典、data_batch、实例化模型等等。

- 其中最重要的就是train_step和inference_step，我们看看inference_step
  
  ```
  def inference_step(self, generator, models, sample, prefix_tokens=None):
     with torch.no_grad():
         return generator.generate(models, sample, prefix_tokens=prefix_tokens)
  ```

- 这里的generator是一个sequence_generator对象，其中生成的部分
  
  ```
  for step in range(max_len + 1):  # one extra step for EOS marker
     # reorder decoder internal states based on the prev choice of beams
     if reorder_state is not None:
         if batch_idxs is not None:
             # update beam indices to take into account removed sentences
             corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
             reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
         model.reorder_incremental_state(reorder_state)
         encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)
  
     lprobs, avg_attn_scores = model.forward_decoder(
         tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
     )
  ```

- 这里做了一层emsemble model的包装，假如我们只有一个decoder模型，那么实际上forward_decoder执行的是
  
  ```
  def _decode_one(
     self, tokens, model, encoder_out, incremental_states, log_probs,
     temperature=1.,
  ):
     if self.incremental_states is not None:
         decoder_out = list(model.forward_decoder(
             tokens,
             encoder_out=encoder_out,
             incremental_state=self.incremental_states[model],
         ))
     else:
         decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
  ```

- 这里可以看到是用incremental decoding逐步解码出句子

# Reorder

- 更多的详细信息可以参考这篇博文，写的非常好，甚至被官方钦点加入了代码注释里[understanding-incremental-decoding-in-fairseq](http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/)
- 还有一点，就是decoder中的reorder，在这篇博文里也有提到。
- 在推断时和训练不同的另一点就是beam search。因此我们不止维护一个缓存队列，而是beam_size个队列。
- 那么在挑选第i个词的时候，第k个beam缓存队列的存的输入token可能是第i-1个位置时第j个beam缓存队列beam search出来的，因此需要重新排序保证一致。

{% endlang_content %}

<script src="https://giscus.app/client.js"
        data-repo="thinkwee/thinkwee.github.io"
        data-repo-id="MDEwOlJlcG9zaXRvcnk3OTYxNjMwOA=="
        data-category="Announcements"
        data-category-id="DIC_kwDOBL7ZNM4CkozI"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="light"
        data-lang="en"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>