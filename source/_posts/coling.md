---
title: Notes for Computational Linguistics
date: 2018-11-16 10:15:34
tags:
  - math
  - machine learning
  - theory
  - nlp
categories:
  - ML
author: Thinkwee
mathjax: true
html: true
---

Course Notes on Computational Linguistics, Reference Textbook: Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition.


<!--more-->

{% language_switch %}

{% lang_content en %}

Chapter 2: Regular Expressions and Automata
===========================================

*   Regular Expressions: A tool used for finding substrings that match a specific pattern or for defining a language in a standardized form, this chapter mainly discusses its function in finding substrings. Regular expressions represent some string sets in an algebraic form.
*   Regular expressions receive a pattern and then search for substrings that match this pattern throughout the corpus, and this function can be realized through the design of a finite state automaton.
*   A string is viewed as a sequence of symbols, where all characters, numbers, spaces, tabs, punctuation marks, and whitespace are considered symbols.

Basic Regular Expression Patterns
---------------------------------

*   Using double slashes to indicate the beginning and end of regular expressions (in Perl's format)
    *   Find substring, case-sensitive: /woodchuck/-> woodchuck
    *   \[One of them is represented by square brackets, or: /\[Ww\]oodchuck/ -> woodchuck or Woodchuck\]
    *   \[±\], take or within the range: /\[2-5\]/->/\[2345\]
    *   Insertion symbols are placed after the left square bracket, representing all symbols that do not appear after them in the pattern, i.e., the negation: /^Ss/ -> neither uppercase S nor lowercase s
    *   Question mark represents the possibility of the previous symbol appearing once or not at all: /colou?r/ -> color or colour
    *   Asterisks represent multiple occurrences or non-occurrences of the preceding symbol: /ba\*/ -> b or ba or baa or baaa......
    *   The plus sign indicates that the preceding symbol appears at least once: /ba+/ -> ba or baa or baaa.......
    *   Decimal point represents a wildcard, matching any symbol except the return character: /beg.n/->begin or begun or beg’n or .......
    *   Anchor symbol, used to represent a substring at a specific location; the insertion symbol represents the beginning of a line, the dollar symbol represents the end of a line; \\b represents a word boundary, \\B represents a non-word boundary; Perl defines a word as a sequence of digits, underscores, or letters, and symbols not included in this are considered as word boundaries.

Extraction, combination, and priority
-------------------------------------

*   Using vertical bars to represent disjunction, or: /cat|dog/ -> cat or dog
*   (gupp(y|ies))/->guppy or guppies
*   Priority: Round brackets > Counters > Sequences and Anchors > Disjunction operator

Advanced Operators
------------------

*   Any number
*   Any non-numeric character
*   Any letter, number, or space
*   Opposite to \\w
*   \\s: blank area
*   S: Opposite to \\s
*   {n}: The preceding pattern appears n times
*   {n,m}: The preceding pattern appears n to m times
*   {n,} : The preceding pattern appears at least n times
*   : newline
*   Tab character

Replacement, Register
---------------------

*   Replace s/A/B/: Replace A with B
*   s/(A)/<\\1>/: Using the numeric operator \\1 to refer to A, placing angle brackets on both sides of A
*   In searches, numerical operators can also be used to represent the content within parentheses, and multiple operators can represent multiple contents within parentheses
*   Here, the numerical operator acts as a register

Finite State Automaton
----------------------

*   Finite state automata and regular expressions are mutually symmetric, and regular expressions are a method to characterize regular languages. Regular expressions, regular grammars, and finite automata are all forms of expressing regular languages. FSA is represented by a directed graph, where circles or dots represent states, arrows or arcs represent state transitions, and double circles represent final states. The state machine diagram below illustrates the recognition of the regular expression /baa+/: ![FoVj3V.png](https://s2.ax1x.com/2019/01/03/FoVj3V.png) 
*   The finite state machine starts from the initial state, reads in symbols sequentially, and if the conditions are met, it performs a state transition. If the sequence of read-in symbols matches the pattern, the finite state machine can reach the final state; if the sequence of symbols does not match the pattern, or if the automaton gets stuck in a non-final state, it is said that the automaton has rejected this input.
*   Another representation is the state transition table: ![FoVqNn.png](https://s2.ax1x.com/2019/01/03/FoVqNn.png) 
*   A finite automaton can be defined by 5 parameters:
    *   Finite set of states {q\_i}
    *   Finite input alphabet
    *   Initial State
    *   Ultimate State Set
    *   The transition function or transition matrix between states, is a relationship from $Q × \Sigma$ to $2^Q$
*   The automaton described above is deterministic, i.e., a DFA, which always knows how to perform state transitions based on the lookup table when the states recorded in the state transition table are known. The algorithm is as follows: given the input and the automaton model, the algorithm determines whether the input is accepted by the state machine: ![FoZpB4.png](https://s2.ax1x.com/2019/01/03/FoZpB4.png) 
*   When an unlisted state occurs, the automaton will malfunction, and a failure state can be added to handle these situations.

Formal language
---------------

*   Formal language is a model that can and only generate and recognize some symbol strings of a certain language that satisfy the definition of formal language. Formal language is a special type of regular language. Formal languages are usually used to simulate certain parts of natural languages. Taking the example /baa+!/ for instance, let the corresponding automaton model be m, the input symbol table be $\Sigma = {a,b,!}$ , and $L(m)$ represents the formal language described by m, which is an infinite set ${baa!,baaa!,baaaa!,…}$ .

Non-deterministic finite automaton
----------------------------------

*   Non-deterministic Finite Automaton (NFSA), by slightly modifying the previous example, moving the self-loop to state 2, it becomes an NFSA, because at this point, in state 2, with input a, there are two possible transitions, and the automaton cannot determine the transition path: ![FoVLhq.png](https://s2.ax1x.com/2019/01/03/FoVLhq.png) 
*   Another form of NFSA involves the introduction of $\epsilon$ transitions, which means that transitions can be made without the need for an input symbol, as shown in the figure below, where at state 3, it is still uncertain how to proceed with the transition: ![FoVX90.png](https://s2.ax1x.com/2019/01/03/FoVX90.png) 
*   At the NFSA, when faced with a transfer choice, the automaton may make an erroneous choice, and there are three solutions to this:
    *   Rollback: Mark this state, and revert to this state after confirming an error in the selection
    *   Prospective: Looking ahead in the input to assist in making choices
    *   Parallel: Performing all possible transfers in parallel
*   In automata, the states that need to be marked when using the backtracking algorithm are called search states, which include two parts: state nodes and input positions. For NFSA, the state transition table also undergoes corresponding changes, as shown in the figure, with the addition of a column representing the $\epsilon$ transition, $\epsilon$ , and the transition can move to multiple states: ![FoZE36.png](https://s2.ax1x.com/2019/01/03/FoZE36.png) 
*   Adopting the fallback strategy, the algorithm for the nondeterministic automaton is as follows, which is a search algorithm: ![FoZSuF.png](https://s2.ax1x.com/2019/01/03/FoZSuF.png) 
*   值
*   The subfunction ACCEPT-STATE accepts a search state, determining whether to accept it, and the accepted search state should be a tuple of the final state and the input end position.
*   Algorithm uses an agenda (process table) to record all search states, initially including only the initial search state, i.e., the initial state node of the automaton and the input start. After that, it continuously loops, extracting search states from the agenda, first calling ACCEPT-STATE to determine if the search is successful, and then calling GENERATE-NEW-STATES to generate new search states and add them to the agenda. The loop continues until the search is successful or the agenda is empty (all possible transitions have been attempted and failed) and returns a rejection.
*   The NFSA algorithm is a state space search that can improve search efficiency by changing the order of search states, for example, by using a stack to implement the process table for depth-first search (DFS); or by using a queue to implement the process table for breadth-first search (BFS).
*   For any NFSA, there exists a completely equivalent DFSA.

Regular Languages and NFSA
--------------------------

*   The alphabet \\(\\sum\\) is defined as the set of all input symbols; the empty symbol string $\epsilon$ , the empty symbol string does not contain in the alphabet; the empty set ∅. The class (or regular set) of regular languages over \\(\\sum\\) can be defined as follows:
    *   The empty set is a regular language
    *   ∀a ∈ $\sum$ ∪ $\epsilon$ , {a} is a formal language
    *   If $L_1$ and $L_2$ are regular languages, then:
    *   Concatenation of $L_1$ and $L_2$ is regular language
    *   The conjunction and disjunction of $L_1$ and $L_2$ are also regular languages
    *   The Kleene closure of $L_1$ is also a regular language, i.e., $L_1$
*   Three basic operators of regular languages: concatenation, conjunction and disjunction, and Kleene closure. Any regular expression can be written in the form that uses only these three basic operators.
*   Regular languages are also closed under the following operations ( $L_1$ and $L_2$ are both regular languages):
    *   The intersection of the symbol string sets of $L_1$ and $L_2$ also constitutes a regular language
    *   The difference set of the symbol sequences of $L_1$ and $L_2$ also constitutes a regular language
    *   The language consisting of sets not in the symbol string collection of $L_1$ is also a regular language
    *   The set of reverses of all symbol strings constitutes a regular language
*   It can be proven that regular expressions and automata are equivalent. A method to prove that any regular expression can be constructed into a corresponding automaton is to, according to the definition of regular languages, construct basic automata representing the single symbol a in $\epsilon$ , ∅, and $\sum$ , and then represent the three basic operators as operations on the automata, inductively applying these operations on the basic automata to obtain new basic automata, thus constructing an automaton that satisfies any regular expression, as shown in the following figure: ![FoVxjU.png](https://s2.ax1x.com/2019/01/03/FoVxjU.png) Basic Automaton ![FoZPE9.png](https://s2.ax1x.com/2019/01/03/FoZPE9.png) Concatenation Operator ![FoZ9HJ.png](https://s2.ax1x.com/2019/01/03/FoZ9HJ.png) Kleene Closure Operator ![FoZiNR.png](https://s2.ax1x.com/2019/01/03/FoZiNR.png) Conjunction Disjunction Operator

Chapter 3: Morphology and Finite State Transcription Machines
=============================================================

*   Analysis: Take an input and produce various structures about this input

Introduction to English Morphology
----------------------------------

*   Morphological study analyzes the structure of words, which can be further decomposed into morphemes. Morphemes can be divided into stems and affixes, and affixes can be further categorized into prefixes, infixes, suffixes, and positional affixes.
*   Inflectional morphology: In English, nouns include only two inflectional forms: one affix indicates plural, and one affix indicates possession:
    *   Plural: -s, -es, irregular plural forms
    *   Ownership: -'s, -s'
*   Inflectional changes in verbs include those of regular verbs and irregular verbs:
    *   Rule verbs: main verbs and basic verbs, -s, -ing, -ed,
    *   Irregular verbs
*   Derivative morphology: Derivation combines a stem with a grammatical morpheme to form new words
    *   Nominalization: -ation, -ee, -er, -ness
    *   Derived adjectives: -al, -able, -less

Morphological analysis
----------------------

*   Example: We hope to establish a morphological analyzer that takes a word as input and outputs its stem and related morphological features, as shown in the table below; our goal is to produce the second and fourth columns: ![FoZA9x.png](https://s2.ax1x.com/2019/01/03/FoZA9x.png) 
*   We at least need:
    *   Lexicon: List of stems and affixes and their basic information
    *   Morphotactic rules: What morphemes follow what morphemes
    *   Orthographic rule: Changes in spelling rules during morpheme combination
*   Generally, word lists are not constructed directly but are designed based on morphological order rules to generate words by inflecting stems. For example, a simple automaton for pluralizing nouns is shown as follows: ![FoZmuD.png](https://s2.ax1x.com/2019/01/03/FoZmuD.png) 
*   reg-noun represents the regular noun, which can be pluralized by adding an "s," and it ignores irregular singular nouns (irreg-sg-noun) and irregular plural nouns (irreg-pl-noun). Another automaton for simulating the inflectional changes of modal verbs is shown below: ![FoZQUA.png](https://s2.ax1x.com/2019/01/03/FoZQUA.png) 
*   A method for using FSA to solve morphological recognition problems (determining whether an input symbol string is legal) is to subdivide state transitions to the letter level, but there will still be some issues: ![FoZZjO.png](https://s2.ax1x.com/2019/01/03/FoZZjO.png) 

Finite State Transcription Machine
----------------------------------

*   Double-layer morphology: Representing a word as a lexical layer and a surface layer, the lexical layer indicates the simple adjacency (concatenation) of morphemes, and the surface layer represents the actual final spelling of the word. A finite state transducer is a finite state automaton, but it implements transcription, achieving correspondence between the lexical layer and the surface layer. It has two inputs, producing and recognizing string pairs, and each state transition arc has two labels, representing the two inputs. ![FoZVgK.png](https://s2.ax1x.com/2019/01/03/FoZVgK.png) 
*   From four perspectives to view FST:
    *   As a recognizer: The FST accepts a pair of strings as input and outputs acceptance if the pair of strings is in the string pairs of the language, otherwise it rejects
    *   As a generator: generating string pairs of language
    *   As a translator: Read in a string, output another
    *   As an associative: calculating the relationship between two sets
*   Define finite state transducer:
    *   A limited state set for state {q_i}
    *   Finite input alphabet
    *   Δ: Finite output symbol alphabet
    *   Initial State
    *   Ultimate State Set
    *   Transition function or transition matrix between states, a relation from Q×Σ to 2^Q, where q is the state and w is the string, returning the new state set
    *   Output function: Given each state and input, returns the set of possible output strings, which is a relation from $Q × \Sigma$ to $2^∆$
*   In FST, the elements of the alphabet are not single symbols, but symbol pairs, known as feasible pairs. Analogous to FSA and regular languages, FST and regular relations are isomorphic, closed under the union operation, and generally not closed under the difference, complement, and intersection operations.
*   Additionally, FST,
    *   On the inverse (inverse of the inverse) closure, the inverse is used to facilitate the transformation from an FST as an analyzer to an FST as a generator
    *   On composite (nested) closures, used to replace multiple transcribing machines with a more complex transcribing machine.
*   Transcription machines are generally non-deterministic; if the search algorithm of a finite state automaton (FSA) is used, it will be very slow, and if a non-deterministic to deterministic conversion algorithm is used, some finite state transducers (FSTs) themselves cannot be converted to be deterministic.
*   Sequential transducer is a deterministic transducer with input, where each state transition is determined after the given state and input, unlike the FST in the figure above, where state 0 has two state transitions upon input b (transferring to the same state but with different outputs). Sequential transducer can use the symbol $\epsilon$ , but it can only be added to the output string, not the input string, as shown in the following figure: ![FoZuHH.png](https://s2.ax1x.com/2019/01/03/FoZuHH.png) 
*   The output of a sequential transducer is not necessarily sequential, that is, different transitions from the same state may produce the same output. Therefore, the inverse of a sequential transducer is not necessarily a sequential transducer. Hence, when defining a sequential transducer, direction must be defined, and the transition function and output function need to be slightly modified, with the output space reduced to Q and ∆.
*   A generalized form of the sequential transcription machine is the concurrent transcription machine, which outputs an additional string in the final state, appended to the string already output. Sequential and concurrent transcription machines are highly efficient, and there are effective algorithms for their determinization and minimization, making them very important. The P-concurrent transcription machine can resolve ambiguity issues on this basis.

Using a finite state transducer for morphological analysis
----------------------------------------------------------

*   Viewing words as the relationship between the lexical layer and the surface layer, as shown in the figure below: ![FoZnDe.png](https://s2.ax1x.com/2019/01/03/FoZnDe.png) 
*   On the basis of the previously defined double-layer morphology, the mapping from a self to itself is defined as a basic pair, represented by a single letter; the symbol ^ represents the morpheme boundary; and the symbol # represents the word boundary. In the task, it is mentioned that it is necessary to output features such as +SG for morphemes, which do not have corresponding output symbols on another output. Therefore, they are mapped to an empty string or boundary symbol. We connect input-output pairs with a colon, or they can also be written above and below an arc. An abstract representation of an English noun plural inflection transducer is shown as follows: ![FoZl4I.png](https://s2.ax1x.com/2019/01/03/FoZl4I.png) 
*   Afterward, we need to update the lexicon so that irregular plural nouns can be parsed into the correct stems: ![FoZMEd.png](https://s2.ax1x.com/2019/01/03/FoZMEd.png) 
*   Afterwards, the abstract transcription machine is concretized into a transcription machine composed of letter transfer arcs, as shown in the figure below, which only displays the concretized part after the irregular plural and singular nouns: ![FoZ3Ct.png](https://s2.ax1x.com/2019/01/03/FoZ3Ct.png) 

Transcription Machines and Orthographic Rules
---------------------------------------------

*   Using spelling rules, also known as orthographic rules, to handle the issue of frequent spelling errors at morpheme boundaries in English.
*   Here are some examples of spelling rules:
    *   Consonant cluster: beg/beggin
    *   Deletion of E: make/making
    *   E insertion: watch/watches
    *   Y's replacement: try/tries
    *   K's insertion: panic/panicked
*   To achieve spelling rules, we introduce an intermediate layer between the lexical and surface layers, taking specific rule-based morpheme concatenation as input and modifying it to produce correct morpheme concatenation as output, for example, the input "fox +N +PL" is transcribed into the intermediate layer, resulting in "fox ^ s#", and then during the second transcription from the intermediate layer to the surface layer, the special morpheme concatenation "x^ and s#" is detected, and an "e" is inserted between the "x" and "s" on the surface layer, yielding "foxes." The following diagram of the transcription machine illustrates this process: ![FoZ88P.png](https://s2.ax1x.com/2019/01/03/FoZ88P.png) 
*   This transducer only considers the positive lexical rule of inserting e when x^ and s# are contiguous
*   Other words can pass normally
*   $Q_0$ represents irrelevant words passing through, indicating an accepted state
*   $Q_1$ represents seeing zsx, serving as an intermediate state, the last z, s, x connected to morphemes are always saved; if other letters appear, it returns to q0, and it can also serve as an accepting state
*   $Q_2$ Represents seeing morphemes connected to z, s, x, followed by four transitions
    *   Received $x$ , $z$ , returned to $q_1$ , that is, believing that reconnected to x, z that may be connected to morphemes
    *   Received $s$ , it is divided into two cases. One is the normal case where e needs to be inserted, in which case it is transferred through $\epsilon$ to $q_3$ and then to $q_4$ . The other is the case where $e$ needs to be inserted from the beginning, reaching $q_5$ , after which it may retreat to $q_1$ , $q_0$ , or $s$ , or return to $q_2$ due to contiguous morphemes. The two cases are uncertain and need to be resolved through search.
    *   Reaching word boundaries and other symbols, returning to $q_0$
    *   It can also function as the passive voice itself

Combine
-------

*   Now, a three-layer structure can be used, combining a transducer for generating morphemes and performing morphological rule correction. From the lexical layer to the intermediate layer, a transducer generates morphemes, and from the intermediate layer to the surface layer, multiple transducers can be used in parallel for morphological rule correction.
*   Two types of transcribing machines can be rewritten as one type when superimposed, at which point the Cartesian product of the state sets of the two types of state machines needs to be calculated, and a state needs to be established for each element in the new set.
*   This three-level structure is reversible, but ambiguity issues may arise during analysis (from the surface to the lexical level), that is, a single word may be analyzed into multiple morpheme combinations. In this case, relying solely on a transcribing machine is insufficient to resolve the ambiguity, and context is needed.

Other Applications (brief introduction)
---------------------------------------

*   FST without a lexicon, PORTER stemmer: Implementing cascading rewriting rules with FST to extract the stems of words.
*   Tokenization and sentence segmentation: A simple English tokenizer can be implemented based on regular expressions, and a simple Chinese tokenizer can be implemented using maxmatch (a greedy search algorithm based on maximum length matching).
*   Spelling Check and Correction: The FST using projection operations can complete the detection of non-word errors, and then correction based on the minimum edit distance (implemented using dynamic programming algorithms) can be performed. Normal word error detection and correction require the assistance of N-gram language models.

How Humans Perform Morphological Processing
-------------------------------------------

*   The study indicates that the human mental lexicon stores a portion of morphological structures, while other structures are not combined in the mental lexicon and require separate extraction and combination. The study elucidates two issues:
    *   Productive morphological forms, particularly inflectional changes, play a role in the human mental lexicon, and the phonological lexicon and the orthographic lexicon may have the same structure.
    *   For example, many properties of morphology, a language processing field, can be applied to the understanding and generation of language.

Chapter 4: N-gram Grammar
=========================

*   Language models are statistical models of word sequences, and the N-gram model is one of them. It predicts the Nth word based on the previous N-1 words, and such conditional probabilities can constitute the joint probability of the entire word sequence (sentence).

Count words in a corpus
-----------------------

*   Difference: Word type, or vocabulary size V, represents the number of different words in the corpus, while tokens, without duplicates, represent the size of the corpus. Some studies suggest that the dictionary size should not be less than the square root of the number of tokens. Non-smooth N-gram grammar model
    
*   Task: Infer the probability of the next word based on previous words: $P(w|h)$ , and calculate the probability of the entire sentence: $P(W)$ .
    
*   The simplest approach is to use the classical probability model, counting the number of occurrences of the segment composed of historical h and current word w in the corpus, and dividing it by the number of occurrences of the historical h segment in the corpus. The probability of the sentence is also generated using a similar method. Drawback: It depends on a large corpus, and the language itself is variable, making the calculation too strict.
    
*   Next, the N-gram grammar model is introduced. Firstly, through the chain rule of probability, the relationship between the conditional probability $P(w|h)$ and the joint probability of the entire sentence $P(W)$ can be obtained:
    
    $$
    P(w_1^n) = P(w_1)P(w_2|w_1)P(w_3|w_1^2)...P(w_n|w_1^{n-1}) \\
    = \prod _{k=1}^n P(w_k|w_1^{k-1}) \\
    $$
    
*   N-gram grammar model relaxes the constraints on conditional probability, making a Markov assumption: the probability of each word is only related to the previous N-1 words, for example, in the bigram grammar model, it is only related to the previous word, using this conditional probability to approximate $P(w|h)$
    
    $$
    P(w_n|w_1^{n-1}) \approx P(w_n|w_{n-1}) \\
    $$
    
*   The conditional probability in the N-gram model is estimated using maximum likelihood, counting the occurrences of various N-gram patterns in the statistical corpus and normalizing them, with a simplification being that, for example, in the case of bigram grammar, the total number of bigrams starting with a given word must equal the count of the unigram of that word:
    
    $$
    P(w_n|w_{n-1}) = \frac {C(w_{n-1}w_n)}{C(w_{n-1})} \\
    $$
    
*   After using N-gram grammar, the chain decomposition of sentence probability becomes easy to calculate, and we can determine whether a sentence contains misspellings by calculating the probabilities of various sentences, or calculate the likelihood of certain sentences appearing in a given context, because N-gram grammar can capture some linguistic features or some usage habits. When there is sufficient corpus, we can use the trigram grammar model to achieve better results.
    

Training set and test set
-------------------------

*   The N-gram grammar model is highly sensitive to the training set. The larger the N in N-gram grammar, the more contextual information it depends on, and the smoother the sentences generated by the N-gram grammar model become. However, they may not be "too smooth" because the N-gram probability matrix is very large and sparse, especially in the case of N being large, such as in a four-gram, where once the first word is generated, the available choices are very few. After generating the second word, the choices become even fewer, often only one choice, resulting in a sentence that is identical to one in the original text with a certain four-gram. Over-reliance on the training set will degrade the model's generalization ability. Therefore, the training set and test set we choose should come from the same subfield.
*   Sometimes, there may be words in the test set that are not present in the training set dictionary, i.e., out-of-vocabulary (OOV) words. In open dictionary systems, we first fix the size of the dictionary and replace all OOV words with special symbols before training.

Evaluation of N-gram Grammar Models: Perplexity
-----------------------------------------------

*   The evaluation of models is divided into two types: external evaluation and internal evaluation. External evaluation is an end-to-end evaluation that examines whether the improvement of a certain module has improved the overall effectiveness of the model. The purpose of internal evaluation is to quickly measure the potential improvement effect of the module. The potential improvement effect of the internal evaluation does not necessarily lead to an increase in the end-to-end external evaluation, but generally, there is some positive correlation between the two.
    
*   Perplexity (PP) is an intrinsic evaluation method for probabilistic models. The perplexity of a language model on a test set is a function of the probabilities assigned by the language model to the test set. Taking binary grammar as an example, the perplexity on the test set is:
    
    $$
    PP(W) = \sqrt[n]{\prod _{i=1}^N \frac {1}{P(w_i|w_{i-1})}} \\
    $$
    
*   The higher the probability, the lower the perplexity. Two interpretations of perplexity:
    
    *   Weighted average branching factor: The branching factor refers to the number of words that can follow any preceding text. It is obvious that if our model has learned nothing, any word in the test set can follow any preceding text, resulting in a high branching factor and high perplexity; conversely, if our model has learned specific rules, words are restricted to follow some specified preceding texts, and the perplexity decreases. Perplexity uses probability-weighted branching factor, and the size of the branching factor remains unchanged before and after the model learns, so "morning" can still follow any preceding text, but the probability of it following "good" increases, thus it is a weighted branching factor.
        
    *   Entropy: For a language sequence, we define the entropy of a sequence as: 
        $$
        H(w_1,w_2,…,w_n )=-\sum _{W_1^n \in L} p(W_1^n) \log ⁡p(W_1^n)
        $$
         which is the sum of the entropies of all prefix sub-sequences within this sequence, and its mean is the entropy rate of the sequence. To calculate the entropy of the entire language, assuming the language is a stochastic process that generates word sequences, with the word sequence being infinitely long, then its entropy rate is: 
        $$
        H(L)=\lim _{n \rightarrow \infty}⁡ \frac 1n H(w_1,w_2,…,w_n) =\lim _{n \rightarrow \infty} -⁡\frac 1n \sum _{W \in L} p(W_1^n)  \log ⁡p(W_1^n)
        $$
         According to the Shannon-McMillan-Breiman theorem, as n approaches infinity, if the language is both stationary and regular, the entropy of the sum of these substrings can be replaced by the maximum substring, where the replacement refers to the probability of the maximum substring calculated after the log, while the probability before the log remains the probability of each substring? If so, the logarithm of the probability of the maximum substring is proposed, and the sum of probabilities of all sub-strings is obtained: 
        $$
        H(L)=\lim _{n \rightarrow \infty} -⁡ \frac 1n \log ⁡p(w_1,w_2,…,w_n)
        $$
         Cross-entropy can measure the distance between the probability distribution generated by our model and the specified probability distribution, and we hope that the probability distribution generated by the model is as close as possible to the true distribution, i.e., the cross-entropy is small. Specifically, it measures the cross-entropy of the probabilities of generating the same language sequence by the model m trained and the ideal model p: 
        $$
        H(p,m) = \lim _{n \rightarrow \infty}⁡ - \frac 1n \sum _{W \in L} p(W_1^n) \log⁡ m(W_1^n)
        $$
         However, we do not know the ideal distribution p. At this point, according to the previous Shannon-McMillan-Breiman theorem, we obtain the cross-entropy of a sequence that only contains one probability distribution (?): 
        $$
        H(p,m)=\lim _{n \rightarrow \infty}⁡ - \frac 1n \log⁡ m(W_1^n)
        $$
         On the test data, since we do not have infinitely long sequences, we approximate the cross-entropy of the infinitely long sequence using the cross-entropy of finite-length sequences. Perplexity is the exponential operation of this (approximate? Only containing one probability distribution?) cross-entropy:
        
        $$
        Perplexity(W) = 2^{H(W)} \\
        = P(w_1 w_2 ... w_N)^{\frac {-1}{N}} \\
        = \sqrt[n]{\frac {1}{P(w_1 w_2 ... w_N)}} \\
        = \sqrt[n]{\prod _{i=1}^N \frac {1}{P(w_i | w_1 ... w_{i-1})}} \\
        $$
        

Smooth
------

*   Because the N-gram model depends on corpus, generally speaking, the higher the N in the N-gram model, the sparser the data provided by the corpus. In this case, the N-gram model performs poorly in estimating grammatical counts that are very small, and if a sentence in the test set contains an N-gram that did not appear in the training set, we cannot use perplexity for evaluation. Therefore, we use smoothing as an improvement method to make the maximum likelihood estimation of the N-gram model adaptable to these situations with 0 probability.
*   Next, two types of smoothing are introduced:
    *   Laplace smoothing (add 1 smoothing)
    *   Good-Turing discounting method

### Laplace Smoothing

*   Add 1 smoothing is to add 1 to each count before calculating probability normalization, correspondingly, the denominator in normalization is increased by the size of the dictionary:
    
    $$
    P_{Laplace}(w_i) = \frac {c_i + 1}{N+V} \\
    $$
    
*   To demonstrate the effect of smoothing, an adjusted count $c^{*}$ is introduced, and the smoothed probability is written in the same form as before the smoothing:
    
    $$
    P_{Laplace} (w_i) = \frac {(C_i^{*})}{N} \\
    C_i^{*} = \frac {(C_i+1)N}{(N+V)} \\
    $$
    
*   An approach to viewing smoothness is: discount each non-zero count, allocate some probability to the zero count, and define relative discount $d_c$ (defined on non-zero counts)
    
    $$
    d_c = \frac {c^{*}} {c}
    $$
    
*   The change in word count before and after the discount is represented by $d_c$ . After smoothing, for non-zero counts, the count increases when $C_i < \frac NV$ , otherwise it decreases. The higher the count, the greater the discount, with less increase (more decrease). When there are many zero counts, N/V is smaller, and in this case, most non-zero counts will decrease, and by a larger amount.
    
*   And the 0 count was not affected by the discount. Therefore, after a round of growth at different levels, the normalized result is that the non-zero counts shared some probability with the 0 count. Written in the form of adjusted counts, it means that the non-zero counts decrease in value, and the 0 count changes (usually decreases) in value (but not the decrease is equal to the increase). The book provides an example, and the figure below shows the counts after binary grammar smoothing of a part of the corpus: ![FoZNDg.png](https://s2.ax1x.com/2019/01/03/FoZNDg.png) If the table is written in the form of adjusted counts: ![FoZtKS.png](https://s2.ax1x.com/2019/01/03/FoZtKS.png) 
    
*   It can be seen that the original 0 count (blue) increases from 0, while others decrease, for example, "<i>" decreases from 827 to 527, from 608 to 238.</i>
    
*   When the count of 0s is many, the reduction in the count of non-0s is significant, and a decimal less than 1, $\delta$ , can be used in place of 1, i.e., adding $\delta$ for smoothing. Typically, this $\delta$ varies dynamically.
    

### GT Discounting Method

*   Similar to the Good-Turing discounting method, the Witten-Bell discounting method, and the Kneyser-Ney smoothing method, their basic motivation is to estimate the count of never-seen items using the count of items that appear only once. Items that appear only once are called singletons or hapax legomena. The Good-Turing discounting method uses the frequency of singletons to estimate the 0-count bigram.
    
*   Define N\_c as the total number of N-gram grammars that appear c times (not the total number multiplied by c), and call it the frequency c frequency. The maximum likelihood estimate of c in N\_c is c. This is equivalent to dividing the N-gram grammar into multiple buckets according to their frequency of occurrence, and the GT discounting method uses the maximum likelihood estimate of the probability of the grammar in the c+1 bucket to reestimate the probability of the grammar in the c bucket. Therefore, after the GT estimation, the c obtained from the maximum likelihood estimate is replaced with:
    
    $$
    c^{*}=(c+1) \frac {N_{c+1}}{N_c}
    $$
    
*   After calculating the probability of a certain N-gram:
    
    *   Never appeared: $P_{GT}^{*}=\frac{N_1}{N}$ . Here, N represents the total number of N-gram grammars $(\sum _i N_i * i)$ . Assuming that $N_0$ is known, this expression indicates that when calculating the probability of a specific unknown N-gram grammar, it should also be divided by $N_0$ .
    *   $P_{GT}^{*} = \frac{c^{*}}{N}$ has appeared (known count):
*   Thus calculated, some probabilities of $N_1$ are transferred to $N_0$ . The GT discounting method assumes that all N-gram probability distributions satisfy the binomial distribution, and assumes that we know $N_0$ , taking the bigram grammar as an example:
    
    $$
    N_0 = V^2 - \sum _{i>0} N_i \\
    $$
    
*   Other considerations:
    
    *   Some $N_c$ are 0, in which case we cannot use these $N_c$ to calculate the smoothed c. In this situation, we directly abandon smoothing, let $c^{*} = c$ , and then calculate a logarithmic linear mapping, $log⁡(N_c) = a + b \log(c)$ , based on the normal data. Substitute the abandoned smoothing c and use its inverse to calculate the $N_c$ with a count of 0, so that these $N_c$ have values and do not affect the calculation of higher-order c.
        
    *   Smooth only the smaller c's $N_c$ , consider the larger c's $N_c$ sufficiently reliable, set a threshold k, and calculate the $N_c$ of $c < k$
        
        $$
        c^{*} = \frac {(c+1) \frac {N_c+1}{N_c} - c \frac {(k+1) N_{k+1} }{N_1} } {1- \frac {(k+1)N_{k+1}} {N_1}} \\
        $$
        
    *   When calculating for smaller c such as c=1, also treat it as the case of c=0 for smoothing
        
*   An example: ![FoZGgf.png](https://s2.ax1x.com/2019/01/03/FoZGgf.png) 
    

Interpolation and Regression
----------------------------

*   The above smoothing only considers how to transfer probabilities to grammatical units with a count of 0. For conditional probability $p(w|h)$ , we can also adopt a similar idea: if there is no ternary grammar to help calculate $p(w_n |w_{n-1} w_{n-2})$ , then a grammar with a lower order, $p(w_n |w_{n-1})$ , can be used to assist in the calculation. There are two options:
    *   Regression: High-order grammar alternative to the 0-count of low-order grammar
    *   Interpolation: Weighted estimation of higher-order grammar using lower-order grammar
*   In the Katz back-off, we use GT discounting as part of the method: GT discounting tells us how much probability can be extracted from the known grammar, while Katz back-off tells us how to distribute these extracted probabilities to the unknown grammar. In the previous GT discounting method, we distributed the extracted probabilities evenly among each unknown grammar, while Katz back-off relies on the information of low-order grammar to distribute: ![FoZJv8.png](https://s2.ax1x.com/2019/01/03/FoZJv8.png) 
*   The probability obtained after the discount is denoted as $P^{*}$ ; \\alpha is the normalization coefficient, ensuring that the probability allocated is equal to the probability obtained from the unknown grammar allocation.
*   Interpolation is obtained by weighted summation using low-order grammatical probabilities to derive the unknown high-order grammatical probabilities: ![FoZUbQ.png](https://s2.ax1x.com/2019/01/03/FoZUbQ.png) 
*   The weighted coefficients can also be dynamically calculated through context. There are two methods for calculating specific coefficients:
    *   Attempt various coefficient combinations, using the one that performs best on the validation set
    *   Viewing coefficients as latent variables in a probabilistic generative model and inferring them using the EM algorithm

Actual Issue: Tools and Data Formats
------------------------------------

*   In language model computation, logarithms of probabilities are generally used for calculation, for two reasons: to prevent numerical underflow; and because taking logarithms converts multiplicative accumulation into additive accumulation, thereby speeding up the computation.
*   The N-gram grammar model generally uses ARPA format. ARPA format files consist of some header information and lists of various N-gram grammars, which include all grammars, probabilities, and normalized coefficients for the N-gram grammars. Only low-order grammars that can be called high-order grammar prefixes can be utilized in the backoff and have normalized coefficients.
*   Two toolkits for calculating N-gram grammar models: SRILM toolkit and Cambridge-CMU toolkit

Advanced Issues in Language Modeling
------------------------------------

### Advanced Smoothing Method: Kneser-Ney Smoothing

*   Noticing that in the GT discounting method, the estimated c value after discounting is approximately one constant d more than the c value obtained from maximum likelihood estimation. The absolute discounting method takes this into account, subtracting this d from each count: ![FoZwUs.png](https://s2.ax1x.com/2019/01/03/FoZwUs.png) 
*   Kneser-Ney smoothing incorporates this perspective and also considers continuity: words that appear in different contexts are more likely to appear after new contexts, and when backtracking, we should prioritize such words that appear in multiple context environments rather than those that occur frequently but only in specific contexts. ![FoZdEj.png](https://s2.ax1x.com/2019/01/03/FoZdEj.png) 
*   In Kneser-Ney, the interpolation method can achieve better results than the back-off method: ![FoZ05n.png](https://s2.ax1x.com/2019/01/03/FoZ05n.png) 

### Based on Categorization N-gram Grammar

*   This method is designed to address the sparsity of training data. For example, in IBM clustering, each word can only belong to one category; for instance, in the case of bigram grammar, the calculation of the conditional probability of a bigram grammar becomes the conditional probability of a word given the category of the preceding context, which can also be further decomposed into the conditional probabilities of two categories multiplied by the conditional probability of a word given its category.
    
    $$
    p(w_i│w_{i-1} ) \approx p(w_i│c_{i-1} ) = p(c_i |c_{i-1}) \cdot p(w_i |c_i)
    $$
    

### Language Model Adaptation and Network Applications

*   Adaptation refers to training language models on large, broad corpora and further improving them on language models in small, specialized domains. The web is an important source of large corpora. In practical applications, it is impossible to search for every grammar and count all grammatical occurrences on all pages retrieved. We use the number of pages retrieved to approximate the count.

### Utilizing longer-distance contextual information

* Usually we use bigram and trigram grammar models, but larger N can also be used to capture more contextual information. To capture longer-distance contextual information, there are several methods:
    *   N-gram Model Based on Caching Mechanism
    *   Based on topic modeling, a N-gram grammar model is applied to different topic modeling language models, followed by weighted summation
    *   Not necessarily using adjacent context information, such as skip N-grams, or not necessarily using fixed-length context information, such as variable-length N-grams

Chapter 16: The Complexity of Language
======================================

Chomsky hierarchy
-----------------

*   Chomsky's hierarchy reflects the implicational relationships between grammars described by different formalization methods, with stronger generative capacity or more complex grammars on the outer layers. From the outside to the inside, the constraints added to the rewrite grammar rules increase, and the generative capacity of the language gradually diminishes. ![FoZXad.png](https://s2.ax1x.com/2019/01/03/FoZXad.png) 
*   Five grammatical rules and application examples corresponding to each: ![Foepxf.png](https://s2.ax1x.com/2019/01/03/Foepxf.png) 
    *   0-type grammar: By rule, there is only one restriction, that is, the left-hand side of the rule cannot be an empty string. 0-type grammar characterizes recursively enumerable languages.
    *   Context-related grammar: The non-terminal symbol A between the contexts \\alpha and \\beta can be rewritten as any non-empty symbol string
    *   Temperate context-related grammar
    *   Context-free grammar: Any single non-terminal symbol can be rewritten as a string consisting of terminal and non-terminal symbols, or it can be rewritten as an empty string
    *   Regular Grammar: It can be right-linear or left-linear. Taking right-linear as an example, a non-terminal symbol can be rewritten as another non-terminal symbol with several terminal symbols added to the left, and right-linear continuously generates terminal symbols on the left side of the string.

Is Natural Language Regular
---------------------------

*   The ability to determine whether a language is regular allows us to understand which level of grammar should be used to describe a language, and this question can help us understand certain formal characteristics of different aspects of natural language.
*   P pumping lemma: Used to prove that a language is not a regular language.
    *   If a language can be described by a finite state automaton, there corresponds to the automaton a memory constraint quantity. This constraint quantity does not increase significantly for different symbol strings, as the number of states is fixed; longer symbol strings should be produced through transitions between states rather than by increasing the number of states. Therefore, this memory quantity does not necessarily scale proportionally with the length of the input.
    *   If a regular language can describe arbitrarily long symbol sequences, more than the number of states in an automaton, then there must be cycles in the automaton. ![FoZxPI.png](https://s2.ax1x.com/2019/01/03/FoZxPI.png) 
*   As shown in the figure of the automaton, it can express xyz, xyyz, xyyyz ..., of course, the infinitely long y sequence in the middle can also be "sucked out," expressing xz. The principle of suction is described as follows:
*   Let L be a finite regular language, then there exist symbol strings x, y, z such that for any n ≥ 0, y ≠ $\epsilon$ , and xy^n z ∈ L
*   If a language is regular, there exists a string y that can be appropriately "absorbed." This theorem is a necessary but not sufficient condition for a language to be regular.
*   Some scholars have proven that English is not a regular language:
    *   Sentences with mirror properties can be proven not to be regular languages through the principle of suction, and a special subset in English is isomorphic to such sentences with mirror properties.
    *   Another proof is based on certain sentences with a central-nested structure. Such sentences can be obtained by the intersection of English and a certain type of simple regular expression, and it can be shown by the pumping lemma that these sentences are not regular languages. If the intersection of English and regular languages is not a regular language, then English is not a regular language.

Is natural language context-free
--------------------------------

*   Since natural language is not a regular language, we then consider a more lenient constraint: is natural language context-free?
*   Not...

Computational Complexity and Human Language Processing
------------------------------------------------------

*   People find it difficult to process centrally nested sentences because the stack memory used in analysis is limited, and memories at different levels in the stack are prone to confusion.

Chapter 5: Part-of-Speech Tagging
=================================

*   Various expressions: POS (Part of Speech), word classes, morphological classes, lexical tags.
*   The significance of POS lies in:
    *   A wealth of information about words and their contexts can be provided.
    *   The same word has different pronunciations under different parts of speech, so POS can also provide information for speech processing.
    *   Perform stemming, assist information retrieval
*   This chapter introduces three word tagging algorithms:
    *   Rule-based algorithm
    *   Probabilistic algorithms, Hidden Markov Models
    *   Algorithm Based on Transformation

General Word Classes
--------------------

*   POS is divided into closed sets and open sets, with closed sets being relatively stable, such as prepositions, while the words in open sets are continuously dynamically expanded, such as nouns and verbs. The open sets of a specific speaker or a specific corpus may differ, but all speakers of a language and various large-scale corpora may share the same closed set. The words in closed sets are called function words (functional words, function words), which are grammatical words, generally short, and have a high frequency of occurrence.
*   Four open classes: nouns, verbs, adjectives, adverbs.
*   Nouns are defined functionally rather than semantically, therefore nouns generally represent people, places, and things, but neither sufficiently nor necessarily. Define nouns:
    *   With the appearance of determiners
    *   Can the subject be modified by pronouns
    *   词以复数形式出现（即可数名词），物质名词不可数。单数可数名词出现时不能没有冠词
*   Verb, a word indicating action and process, including the forms of third-person singular, non-third-person singular, present continuous, and past participle
*   Adjectives, describing nature and quality
*   Adverb, used for modification; adverbs can modify verbs, verb phrases, and other adverbs.
*   Some closed classes in English:
    *   Prepositions: Appear before noun phrases, indicating relationships
    *   Determiners: related to definiteness Articles: related to definiteness
    *   Pronouns: A brief form of reference to certain noun phrases, entities, or events
    *   Conjunctions: Used for connection and complementation (complementation)
    *   Auxiliary verbs: Mark certain semantic features of the main verb, including: tense, perfective aspect, polarity opposition, modality
    *   Particles: Combine with verbs to form phrasal verbs
    *   Numerals

Word tagging
------------

*   The input of annotation algorithms is a sequence of word symbols and a set of tags, and the output requires each word to be annotated with a single and optimal tag. If each word corresponds to only one part of speech, then according to the existing set of tags, part-of-speech tagging is a simple process of lookup and labeling. However, many words have multiple parts of speech, such as "book," which can be both a noun and a verb, thus requiring disambiguation. Part-of-speech tagging is an important aspect of disambiguation.

Rule-based Part-of-Speech Tagging
---------------------------------

*   Presented the ENGTWOL system, constructed based on a double-layered morphology, establishing separate entries for each word type, and calculating without considering inflectional and derivative forms.
*   The first stage of the annotation algorithm involves using a two-layer transducer to obtain all possible word categories for a given word
*   Afterward, incorrect word classes are excluded by applying constraint rules, which determine which word classes to exclude based on the type of context.

Word Classification Based on Hidden Markov Models
-------------------------------------------------

*   Using Hidden Markov Models for word segmentation is a type of Bayesian inference, where word segmentation is viewed as a sequence classification task. The observation is a word sequence (such as a sentence), and the task is to assign a labeling sequence to this sequence.
    
*   Given a sentence, Bayesian inference aims to select the best sequence among all possible annotated sequences, that is
    
    $$
    {t_1^n} _{best} = {argmax} _{t_1^n}  P(t_1^n |w_1^n)
    $$
    
*   Using Bayes' theorem, it can be transformed into:
    
    $$
    {t_1^n} _{best}={argmax} _{t_1^n}  \frac{P(w_1^n│t_1^n)P(t_1^n)}{P(w_1^n)} = {argmax} _{t_1^n} P(w_1^n│t_1^n)P(t_1^n)
    $$
    
*   The Hidden Markov Model makes two assumptions on this basis
    
    *   The probability of a word's occurrence is only related to the word's part-of-speech tagging, and is unrelated to other words in the context or other tags, thus decomposing the joint probability of the sequence into the product of element probabilities, i.e., P(w\_1^n│t\_1^n) ≈ ∏\_{i=1}^n P(w\_i |t\_i)
    *   A labeled probability is only related to the previous label, similar to the assumption of binary grammar: P(t\_1^n) ≈ ∏\_{i=1}^n P(t\_i |t\_{i-1})
*   Under two assumptions, the best annotated sequence expression after simplification is:
    
    $$
    {t_1^n}_{best} = {argmax} _{t_1^n} P(t_1^n│w_1^n) \approx {argmax} _{t_1^n} \prod _{i=1}^n P(w_i│t_i) P(t_i |t_{i-1})
    $$
    
*   The above probability expression actually decomposes the joint probability of the HMM model into the product of individual transition probabilities, specifically into label transition probabilities (transitions between hidden variables) and word likelihood (transitions from hidden variables to observable variables). Through maximum likelihood estimation, we can calculate these two types of probabilities using the classical probability type method from the annotated corpus:
    
    $$
    P(t_i│t _{i-1} ) = (C(t _{i-1},t_i))/C(t _{i-1} ) \\
    P(w_i│t_i ) = \frac{C(t_i,w_i)}{C(t_i)} \\
    $$
    
*   An example: How the HMM model correctly identifies "race" as a verb rather than a noun in the following sentence:
    
*   The Secretariat is expected to race tomorrow.
    
*   Sketch the HMM models for the two cases where "race" is identified as both a verb and a noun, and you can see that only three transition probabilities differ between the two models, which are marked with bold lines: ![FoZDCq.png](https://s2.ax1x.com/2019/01/03/FoZDCq.png) 
    
*   The HMM word sense disambiguator operates in a global rather than a local manner. We obtain these three transition probabilities by counting in the corpus and then multiplying them together, resulting in the probability of (a) being 843 times that of (b). It is obvious that "race" should be tagged as a verb.
    

Formalized Hidden Markov Model Annotator
----------------------------------------

*   HMM model is an extension of finite automata, specifically a weighted finite automaton, an extension of Markov chains, which allows us to consider observed variables and hidden variables, and consider probabilistic models that include hidden variables. The HMM includes the following components:
    
    *   State set of size N
    *   A: A transfer probability matrix of size N\*N
    *   Observation event set of size T
    *   B: The observation likelihood sequence, also known as emission probability, $b_i (o_t)$ describes the probability of generating observation o\_t from state i
    *   Special initial and final states, without connected observation quantities
*   The probability in A corresponds to the prior $P(w_i│t_i )$ and likelihood $P(t_i |t _{i-1})$ probabilities in each cumulative product term of the previous formula:
    
    $$
    {t_1^n}_{best}={argmax} _{t_1^n} P(t_1^n│w_1^n ) \approx {argmax} _{t_1^n} \prod _{i=1}^n P(w_i│t_i)P(t_i |t _{i-1})
    $$
    

Hidden Markov Model (HMM) Viterbi Algorithm
-------------------------------------------

*   In the HMM model, the task of inferring the hidden variables given the transition probabilities and the observation sequence is called decoding. One algorithm for decoding is the Viterbi algorithm, which is essentially a dynamic programming algorithm, similar to the algorithm previously used to find the minimum edit distance.
*   First, we calculate two matrices A and B from the corpus, that is, the transition probabilities of the model are known. For a given observation sequence, the Viterbi algorithm is executed according to the following steps: ![FoZyvT.png](https://s2.ax1x.com/2019/01/03/FoZyvT.png) 
*   The algorithm maintains a Viterbi probability matrix $(N+2)*T$ with 2 representing the initial and final states. Viterbi\[s,t\] represents the best path probability at state s in step t, while backpointer\[s,t\] corresponds to the previous state that led to the best path, used for backtracking to output the entire best path.
*   The key transition lies in $viterbi[s,t] \leftarrow max _{s^{*}=1}^N⁡ viterbi[s^{*},t-1] * a_{s^{*},s} * b_s (o_t)$ that the optimal path at the current time step is transferred from the optimal paths of various states in the previous time step. The path with the maximum product of the probability of the optimal path in the previous step and the transition probability is chosen as the optimal path at the current time step. From the perspective of dynamic programming, the optimal path of length t must be selected from the optimal paths of length t-1, otherwise, it is certain that a better solution can be obtained by transferring from another path with a higher probability. This limits the possibilities of generating the optimal path and reduces the computational amount.

Extending the HMM algorithm to trigram grammar
----------------------------------------------

*   Modern HMM annotators generally consider a longer history of the preceding context in the annotation of transition probabilities:
    
    $$
    P(t_1^n ) \approx \prod_{i=1}^n P(t_i |t _{i-1},t_{i-2})
    $$
    
*   Such a case requires boundary handling at the beginning and end of the sequence. One issue with using trigram grammar is data sparsity: for example, if we have never seen the annotated sequence PRP VB TO in the training set, we cannot compute P(TO|PRP,VB). One solution is linear interpolation:
    
    $$
    P(t_i│t _{i-1} t _{i-2} ) = \lambda _1 P ̂(t_i│t _{i-1} t _{i-2} )+\lambda _2 P ̂(t_i│t _{i-1} )+\lambda _3 P ̂(t_i)
    $$
    
*   Determine the coefficient $\lambda$ using the method of deletion interpolation: ![FoZr80.png](https://s2.ax1x.com/2019/01/03/FoZr80.png) 
    

Transformation-Based Annotation
-------------------------------

*   The method based on transformation combines the advantages of rule-based and probabilistic methods. The method based on transformation still requires rules, but it summarizes rules from the data, which is a supervised learning approach known as Transformation Based Learning (TBL). In the TBL algorithm, the corpus is first annotated with relatively broad rules, then slightly special rules are selected to modify, followed by narrower rules to modify a smaller number of annotations.

How to Apply the TBL Rules
--------------------------

*   Firstly, the most general rule is applied, which is to annotate each word based on probability and select the word class with the highest probability as the annotation. Then, transformation rules are applied, meaning that if a certain condition is met, the previously annotated word class is transformed (corrected) into the correct word class. Subsequently, more stringent transformations are continuously applied, making minor modifications based on the previous transformation.
*   How to Learn the TBL Rules
    *   First, label each word with the most probable tag
    *   Examine each possible transformation, select the transformation that yields the greatest improvement in effect, and here it is necessary to use the correct label for each word to measure the improvement brought by the transformation, therefore it is supervised learning.
    *   According to this selected transformation, re-label the data, repeat step 2 until convergence (the improvement effect is less than a certain threshold)
*   The output of the above process is a sequence of ordered transformations, used to form a labeling process and applied to new corpus. Although all rules can be enumerated, the complexity is too high, so we need to limit the size of the transformation set. The solution is to design a small set of templates (abstract transformations), where each allowed transformation is an instantiation of one of the templates.

Evaluation and Error Analysis
-----------------------------

*   Generally, it is divided into training set, validation set, and test set, with ten-fold cross-validation performed within the training set.
*   Comparing the computational accuracy with the gold standard of human annotation as a measure.
*   The general human performance is used as a ceiling, and the result with the highest probability marked by the one-way grammar is used as the baseline.
*   Through confusion matrices or contingency tables to conduct error analysis. In an N-classification task, the element in the ith row and jth column of an N\*N confusion matrix indicates the proportion of times that the ith class is mistakenly classified as the jth class out of the total number of misclassifications. Some common word types that are easy to misclassify include:
    *   Single nouns, proper nouns, adjectives
    *   Adverbs, particles, prepositions
    *   Verb past tense, verb past participle, adjective

Some other issues in part-of-speech tagging
-------------------------------------------

*   Labeling uncertainty: A word has ambiguity between multiple parts of speech, which is difficult to distinguish. In this case, some annotators allow a word to be labeled with multiple part-of-speech tags. During training and testing, there are three methods to address the issue of multi-labeled words:
    *   Select a label from these candidate labels in some way
    *   Specify a word type during training, and consider the annotation correct as long as any of the candidate word types are marked during testing
    *   View the entire set of uncertain parts of speech as a new complex part of speech
*   Multicomponent words: Before annotation, segmentation is required. Whether some multicomponent words should be segmented into one part, such as whether "New York City" should be divided into three parts or treated as a whole, is also a consideration for various annotation systems.
*   Unknown word: Words not found in dictionaries are called unknown words. For unknown words, the training set cannot provide its likelihood P(w\_i | t\_i), which can be addressed in the following ways:
    *   Predicting POS based solely on contextual information
    *   Estimating the distribution of unknown words using words that appear only once, similar to the Good Turing discounting method
    *   Utilizing spelling information of words with unknown words, morphological information. For example, hyphenation, ed endings, capitalized initials, etc. Subsequently, calculate the likelihood of each feature in the training set, assuming independence among features, and then multiply the likelihoods of features as the likelihood of the unknown word: $P(w_i│t_i )=p(unknown word│t_i ) * p(capital│t_i ) * p(endings/hyph|t_i)$
    *   Maximum Entropy Markov Model
    *   Utilizing the log-linear model

Noise Channel Model
-------------------

*   Bayesian inference is used for annotation, which can be considered an application of a noise channel model. This section introduces how to use the noise channel model to complete the task of spelling correction. Previously, non-word errors could be detected through dictionary lookup and corrected based on the minimum edit distance, but this method is ineffective for real word errors. The noise channel model can correct both types of spelling errors.
*   The motivation of the noise channel model lies in treating a misspelled word as a correctly spelled word that has been distorted and interfered with after passing through a noise channel. We try all possible correct words, input them into the channel, and then compare the word after interference with the misspelled word; the input word that corresponds to the most similar example is considered the correct word. Such noise channel models, such as the previous HMM tagging model, are a special case of Bayesian inference. We observe a misspelled word and hope to find the latent variable (correctly spelled word) that generates this observation, which is to find the maximum a posteriori.
*   Applying the noise channel model to spelling correction: First, assume various types of spelling errors, such as misspelling one, misspelling two, omitting one, etc., then generate all possible corrections, excluding those not existing in the dictionary, and finally, calculate the posterior probabilities separately, selecting the correction with the highest posterior probability. In this process, likelihood needs to be calculated based on local contextual features.
*   Another correction algorithm is a method that improves through iteration: first, assume that the ambiguous matrix for spelling correction is uniformly distributed, then run the correction algorithm based on the ambiguous matrix, and update the ambiguous matrix according to the corrected dataset, repeating the iteration. This iterative algorithm is an EM algorithm.

Contextual spelling correction
------------------------------

*   Correction of actual word spelling errors. To address such tasks, it is necessary to extend the noise channel model: when generating candidate correction words, the word itself and homophones should be included. Subsequently, the correct correction word is selected based on the maximum likelihood of the entire sentence.

Chapter 6: Hidden Markov Models and Maximum Entropy Models
==========================================================

*   Hidden Markov Model is used to solve sequence labeling (sequence classification problem).
*   The maximum entropy method is a classification idea that, under given conditions, the classification should satisfy the minimum restrictions (maximum entropy) and comply with Ockham's Razor principle.
*   The maximum entropy Markov model is an extension of the maximum entropy method to sequence labeling tasks.

Markov Chain
------------

*   Weighted finite automata are an extension of finite automata, where each transition path is assigned a probability as a weight, indicating the possibility of transition along that path. A Markov chain is a special case of a weighted finite state automaton, where the input sequence uniquely determines the sequence of states the automaton will pass through. A Markov chain can only assign probabilities to deterministic sequences.
    
*   We regard the Markov chain as a probabilistic graph model; a Markov chain is determined by the following components:
    
    $$
    Q=q_1 q_2…q_N \\
    A=a_{01} a_{02} … a_{n1} … a_{nn} \\
    q_0,q_F \\
    $$
    
*   Respectively
    
    *   State Set
    *   Transfer probability matrix, where a\_ij represents the probability of transitioning from state i to state j $P(q_j |q_i)$
    *   Special starting and ending states
*   Probability graphs represent states as points in a graph and transitions as edges.
    
*   First-order Markov models make a strong assumption about transitions: the probability of a state is only related to the previous state:
    
    $$
    P(q_i│q_1…q _{i-1} )=P(q_i |q _{i-1})
    $$
    
*   Another representation of Markov chains does not require a starting and ending state:
    
    $$
    \pi = \pi _1,\pi _2 , … , \pi _N \\
    QA={q_x,q_y…} \\
    $$
    
*   Are:
    
    *   The initial probability distribution of the state, the Markov chain starts from state i with probability $\pi _i$
    *   Set QA is a subset of Q, representing a legitimate acceptance state
*   Therefore, the probability of state 1 as the initial state can be written as $a_{01}$ or as $\pi _1$ .
    

Hidden Markov Model
-------------------

*   When the Markov chain is known, we can use it to calculate the probability of an observed sequence. However, the observed sequence may depend on some unobserved hidden variables, and we may be interested in inferring these hidden variables. The Hidden Markov Model allows us to consider both observed variables and hidden variables simultaneously.
*   As defined previously, the Hidden Markov Model:
    *   State set of size N
    *   A: A transfer probability matrix of size N\*N
    *   Observation event set of size T
    *   B: The observation likelihood sequence, also known as emission probability, $b_i (o_t)$ describes the probability of generating observation $o_t$
    *   Special initial and final states, without connected observation quantities
*   Similarly, the Hidden Markov Model can also be represented in a way that does not depend on the initial and final states. The Hidden Markov Model also makes two assumptions, namely the first-order Markov property between hidden states and from hidden states to observations.
*   For Hidden Markov Models, three types of problems need to be addressed:
    *   Likelihood Calculation: Given parameters and an observation sequence, calculate the likelihood $P(O|\lambda)$
    *   Decoding: Given the known parameters and the observed sequence, to find the hidden state sequence
    *   Learning: Solving model parameters given the observed sequence and the set of hidden states

Computing Likelihood: Forward Algorithm
---------------------------------------

*   For the Markov chain, as it lacks a transition probability matrix from hidden states to observables, it can be considered as having observables and hidden states being the same. In the Hidden Markov Model, it is not possible to directly calculate the likelihood; we need to know the hidden state sequence.
    
*   Assuming the hidden state sequence is known, the likelihood calculation is:
    
    $$
    P(O│Q) = \prod _{i=1}^T P(o_i |q_i)
    $$
    
*   According to the first-order Markov property of hidden state transitions, the prior of the hidden state can be obtained, multiplied by the likelihood to obtain the joint probability of the observed sequence and the hidden state sequence:
    
    $$
    P(O,Q)=P(O│Q) * P(Q) = \prod _{i=1}^n P(o_i│q_i )  \prod _{i=1}^n P(q_i |q _{i-1})
    $$
    
*   For the joint probability integral over the hidden state sequence, the likelihood of the observed probability can be obtained:
    
    $$
    P(O) = \sum _Q P(O,Q) = \sum _Q P(O|Q)P(Q)
    $$
    
*   This calculation is equivalent to considering all possible hidden states and calculating the likelihood for each possibility from the start to the end of the hidden state sequence. In fact, it is possible to retain the intermediate states of each calculation to reduce redundant computation, which is known as dynamic programming. The dynamic programming algorithm used in the forward calculation of the HMM observation likelihood is called the forward algorithm:
    
    *   Let $\alpha _t (j)$ represent the probability that the latent variable is in state j at the current moment after obtaining the first t observations, with λ being the model parameter:
        
        $$
        \alpha _t (j) = P(o_1,o_2…o_t,q_t=j|\lambda)
        $$
        
    *   This probability value can be calculated based on the \\alpha value of the previous time step, thus avoiding recalculating from scratch each time:
        
        $$
        \alpha _t (j) = \sum _{i=1}^N \alpha _{t-1} (i) a_{ij} b_j (o_t)
        $$
        
    *   Initialization $\alpha _1 (j)$ :
        
        $$
        \alpha _1 (j)=a_{0s} b_s (o_1)
        $$
        
    *   Termination state:
        
        $$
        P(O│\lambda) = \alpha _T (q_F) = \sum _{i=1}^N \alpha _T (i) \alpha _{iF}
        $$
        

Decoding: Viterbi Algorithm
---------------------------

*   The decoding task is to infer the most likely hidden state sequence based on the observed sequence and parameters. The most naive approach is to calculate the likelihood of the observed sequence for each possible hidden state sequence and take the hidden state sequence corresponding to the maximum likelihood. However, this is just like the naive calculation of likelihood methods, with a high time complexity. Similarly, we use dynamic programming to reduce the scale of the solution. A Viterbi algorithm is used during decoding.
    *   Let $v_t (j)$ represent the probability of the current latent state being j, given the known first t observations (1t) and the known first t-1 latent states (0t-1)
        
        $$
        v_t (j)=max _{q_0,q_1,…,q_{t-1}} P(q_0,q_1…q_{t-1},o_1,o_2 … o_t,q_t=j|\lambda)
        $$
        
    *   Among which, we have known the maximum possible hidden state sequence for the first t time steps, which is also obtained through dynamic programming:
        
        $$
        v_t (j)=max _{i=1}^N⁡ v_{t-1} (i) a_{ij} b_j (o_t)
        $$
        
    *   To obtain the optimal hidden state sequence, it is also necessary to record the best choice at each step for easy backtracking to obtain the path:
        
        $$
        {bt}_t (j) = argmax _{i=1}^N v_{t-1} (i) a_{ij} b_j (o_t)
        $$
        
    *   Initialization:
        
        $$
        v_1 (j) = a_{0j} b_j (o_1) \ \  1 \leq j \leq N \\
        {bt}_1 (j) = 0 \\
        $$
        
    *   Termination, separately obtaining the optimal hidden state sequence (with the starting value for backtracking) and its likelihood value:
        
        $$
        P * = v_t (q_F ) = max_{i=1}^N⁡ v_T (i) * a_{i,F} \\
        q_{T*} = {bt}_T (q_F ) = argmax _{i=1}^N v_T (i) * a_{i,F} \\
        $$
        
*   The reason the Viterbi algorithm reduces the time complexity is that it does not compute all the hidden state paths but utilizes the condition that the best path at each time step can only extend from the best path at the previous time step, thereby reducing the number of path candidates and avoiding many unnecessary path computations. Moreover, using the result from the previous step also employs the idea of dynamic programming to reduce the amount of computation.

Training Hidden Markov Models: Forward-Backward Algorithm
---------------------------------------------------------

*   Learning problems refer to the situation where the known observed sequence and the set of hidden states are given, and the model parameters are to be solved for.
    
*   Forward-backward algorithms, also known as the Baum-Welch algorithm, are a special case of the EM algorithm used to estimate the parameters of probabilistic generative models containing latent variables. The algorithm updates the transition probabilities and generation probabilities iteratively until convergence. The BW algorithm uses the ratio of the counts as the latent variables, iteratively updating the transition probability matrix and the generation probability matrix together.
    
*   Consider the learning problem of Markov chains. Markov chains can be regarded as degenerate hidden Markov models, where each hidden variable generates only observations of itself, with a probability of 0 for generating other observations. Therefore, only the transition probabilities need to be learned.
    
*   For Markov chains, the transition probabilities can be statistically estimated through the classical probability model:
    
    $$
    a_{ij} = \frac {Count(i \rightarrow j)} {\sum _{q \in Q} Count(i \rightarrow q)}
    $$
    
*   We can directly calculate the probability because in a Markov chain, we know the current state. For the Hidden Markov Model, we cannot calculate it directly because the hidden state sequence cannot be determined for a given input. The Badum-Welch algorithm uses two simple intuitions to solve this problem:
    
    *   Iterative estimation, first assuming a transition probability and a generation probability, and then deducing better probabilities based on the assumed probabilities
    *   Estimate the forward probability of a certain observation, and distribute this probability across different paths, thereby estimating the probability
*   Firstly, similar to the forward probability, we define the backward probability:
    
    *   Let $\beta _t (i)$ represent the probability that the latent variable is in state i at the current moment after receiving t observations, and $\lambda$ are the model parameters:
        
        $$
        \beta _t (i) = P(o_{t+1},o_{t+2}…o_T,q_t=i|\lambda)
        $$
        
    *   Similar to inductive calculations of backward probability:
        
        $$ \\beta\_t (i) = \\sum \_{j=1}^N a\_{ij} b\_j (o\_{t+1} ) \\beta \_{t+1} (j), \\ \\ 1≤i≤N,1≤t
        
    *   Initialization $$ :
        
        $$
        \alpha _1 (j)
        $$
        
    *   Termination state:
        
        $$
        \beta _T (i)=\alpha _(i,F)
        $$
        
*   Similarly, we hope the classical probability in the Markov chain can help us estimate the transition probabilities:
    
    $$
    P(O│\lambda)=\alpha _t (q_F )=\beta_1 (0)= \sum _{i=1}^N a_{0j} b_j (o_1) \beta _1 (j)
    $$
    
*   How to estimate the count values: We convert the count values of the entire sequence's transition paths into the sum of count values between time steps, with the probability of a specific transition path between time steps being:
    
    $$
    a_{ij}^{*} = \frac {the expected count of transitions from state i to state j}{the expected count of transitions from state i to other states}
    $$
    
*   First, consider the joint probability of all the observed sequences and this transition path (conditioned on parameters $P(q_t=i,q_{t+1}=j)$ is omitted):
    
    $$
    \lambda
    $$
    
*   Observe the following probability graph: ![FoZWVJ.png](https://s2.ax1x.com/2019/01/03/FoZWVJ.png) 
    
*   It can be seen that this joint probability includes three parts:
    
    *   T-moment hidden state i forward probability
    *   Backward probability of the hidden state j at T+1 moment
    *   Probability of state transition between T time and T+1 time, as well as the generation probability of the corresponding observed quantities
*   Therefore, there is:
    
    $$
    P(q_t=i,q_{t+1}=j,O)
    $$
    
*   To obtain the joint probability of transition paths from the joint distribution given the known observation sequence, it is necessary to calculate the probability of the observation sequence, which can be obtained through forward probability or backward probability:
    
    $$
    P(q_t=i,q_{t+1}=j,O)=\alpha _t (i) a_{ij} b_j (o_{t+1} ) \beta _{t+1} (j)
    $$
    
*   Ultimately obtained
    
    $$
    P(O)=\alpha _t (N)=\beta _T (1) = \sum _{j=1}^N \alpha _t (j) \beta_t (j)
    $$
    
*   The sum of all time steps yields the expected count of transitions from state i to state j, thereby further obtaining an estimate of the transition probability:
    
    $$
    ξ_t (i,j)=P(q_t=i,q_{t+1}=j│O) = \frac {(\alpha _t (i) a_{ij} b_j (o_{t+1} ) \beta_{t+1} (j))}{(\alpha _t (N))}
    $$
    
*   Similarly, we also hope to obtain an estimate of the generation probability:
    
    $$
    a_{ij}^{*} = \frac {\sum _{t=1}^{T-1} ξ_t (i,j)}{\sum _{t=1}^{T-1} \sum _{j=1}^{N-1} ξ_t (i,j)}
    $$
    
*   Similarly, the probability of being in the hidden state j at time t is obtained by first calculating the joint distribution and then the conditional distribution:
    
    $$
    b_{j}^{*} (v_k) = \frac {the expected count of observations of symbol v_k in state j}{the expected count of observations of all symbols in state j}
    $$
    
*   The joint probability includes two parts, namely the forward probability and the backward probability of being in state j at time t, thus:
    
    $$
    γ_t (j)=P(q_t=j│O) = \frac {P(q_t=j,O)}{P(O)}
    $$
    
*   Similarly, by summing over all time steps, an estimate of the generation probability is obtained:
    
    $$
    γ_t (j) = \frac {\alpha _t (j) \beta_t (j)}{\alpha _t (N)}
    $$
    
*   These two formulas are calculated under the condition of known forward and backward probabilities, and introduce the intermediate variables (latent variables) (ξ,γ). The motivation for introducing latent variables is to transform the ratio of the expected counts of the estimates of a and b into a ratio of probabilities, and these two latent variables can be represented by a and b. Then, the transition probabilities and generation probabilities are calculated from the latent variables, thus forming an iterative loop, which can be solved using the EM algorithm.
    
    $$
    (\alpha,\beta)
    $$
    
*   E-step:
    
    $$
    a,b→\alpha,\beta→ξ,γ→a,b
    $$
    
*   M-step (What is the goal of maximization):
    
    $$
    γ_t (j) = (\alpha _t (j) \beta_t (j))/(\alpha _t (N)) ξ_t (i,j) \\
    = (\alpha _t (i) a_{ij} b_j (o_{t+1} ) \beta_{t+1} (j))/(\alpha _t (N)) \\
    $$
    
*   Iterative calculations need to be recalculated:
    
    $$
    a _{ij} = (\sum _{t=1}^{T-1}   ξ_t (i,j)  )/(\sum _{t=1}^{T-1} \sum _{j=1}^{N-1}   ξ_t (i,j)  ) \\
    b ̂_j(v_k) = (\sum _{t=1 s.t. O_t=v_k}^T   γ_t (j) )/(\sum _{t=1}^T   γ_t (j) ) \\
    $$
    
*   The initial state of the iteration is important for the EM algorithm, often designed by introducing some external information.
    

Maximum Entropy Model: Background
---------------------------------

*   Another widely known form of the maximum entropy model is multi-logistic regression (Softmax?).
    
*   The maximum entropy model solves classification problems. As a probabilistic classifier, the maximum entropy model can calculate the probability of each sample belonging to every category based on the features of the samples, and then perform classification.
    
*   The maximum entropy model belongs to the exponential family (log-linear) classifier, which calculates classification probabilities by taking the exponential of a linear combination of features:
    
    $$
    \alpha _t (j) = \sum _{i=1}^N   \alpha_{t-1} (i) a_ij b_j (o_t) \\
    \beta_t (i) = \sum _{j=1}^N   a_ij b_j (o_{t+1} ) \beta_{t+1} (j)  \\
    $$
    
*   Z is a normalization coefficient that makes the sum of the generated probabilities equal to 1.
    

Maximum Entropy Modeling
------------------------

*   Extending binary logistic regression to the multi-class problem results in:
    
    $$
    p(c│x)=\frac 1Z exp⁡(\sum _i   weight_i feature_i)
    $$
    
*   Features in speech and language processing are typically binary (whether the feature is present), thus indicator functions are used to represent features
    
    $$
    P(c│x) = \frac {exp⁡(\sum _(i=0)^N   w_ci f_i) } {\sum _{c^{*} in C}   exp⁡(\sum _{i=0}^N   w_{c^{*} i} f_i)  }
    $$
    
*   Noticed that in this model, each class has its independent linear weight w\_c. Compared to hard distribution, the maximum entropy model can provide the probability of being assigned to each class, thus allowing the calculation of the classification probability at each moment, and then the overall classification probability can be obtained, leading to the global optimal classification result. Noticed that unlike support vector machines and other models, the maximum entropy model cannot utilize the combination of features and must manually construct combinations as new features.
    
*   通常使用加上了正则化的最大似然作为优化的目标函数：
    
    $$
    P(c│x) = \frac {exp⁡(\sum _{i=0}^N   w_{c_i} f_i (c,x)) }{\sum _{c^{*} \in C}   exp⁡(\sum _{i=0}^N   w_{c^{*} i} f_i (c^{*},x))  }
    $$
    
*   This regularization is equivalent to adding a zero-mean Gaussian prior to the probability distribution of the weights, where the weights deviate more from the mean, i.e., the larger the weights, the lower their probability.
    
*   Why multi-class logistic regression is a maximum entropy model: The maximum entropy model guarantees that the un 约束 part of the classification should be equally probable under given constraints, for example, under two constraints:
    
    $$
    w ̂={argmax} _w \sum _i   \log P(y^{(i)}│x^{(i) } ) - \alpha \sum _{j=1}^N w_j^2
    $$
    
*   Then, if these two constraints are satisfied, the probability results allocated by the maximum entropy model are:
    
    $$
    P(NN)+P(JJ)+P(NNS)+P(VB)=1 \\
    P(t_i=NN or t_i=NNS)=8/10 \\
    $$
    
*   In the paper "The Equivalence of Logistic Regression and Maximum Entropy Models," it is proven that under the constraint of the balance condition in the generalized linear regression model, the nonlinear activation function that satisfies the maximum entropy distribution is the sigmoid, i.e., logistic regression.
    

Maximum Entropy Markov Model
----------------------------

*   The maximum entropy model can only classify single observations, while the maximum entropy Markov model can extend it to the problem of sequence classification.
    
*   Where does the Maximum Entropy Markov model excel over the Hidden Markov Model? The Hidden Markov Model depends on transition probabilities and generation probabilities for the classification of each observation. If we want to introduce external knowledge during the labeling process, we need to encode this external knowledge into these two types of probabilities, which is inconvenient. The Maximum Entropy Markov model can introduce external knowledge more simply.
    
*   In the Hidden Markov Model, we optimize the likelihood and multiply by the prior to estimate the posterior:
    
    $$
    p(NN)=4/10  \\
    p(JJ)=1/10  \\
    p(NNS)=4/10  \\
    p(VB)=1/10 \\
    $$
    
*   In the maximum entropy hidden Markov model, we directly compute the posterior. Because we directly train the model for classification, that is, the maximum entropy Markov model is a type of discriminative model rather than a generative model:
    
    $$
    T ̂= {argmax}_T ∏_i   P(word_i│tag_i ) ∏_i   P(tag_i│tag _{i-1} )
    $$
    
*   Therefore, in the maximum entropy hidden Markov model, there is no separate modeling of likelihood and prior, but rather the posterior is estimated through a single probability model. The difference between the two is shown in the figure below: ![FoZgrF.png](https://s2.ax1x.com/2019/01/03/FoZgrF.png) 
    
*   Additional features can be more dependent and flexible in the Maximum Entropy Markov Model, as shown in the following figure: ![FoZcKU.png](https://s2.ax1x.com/2019/01/03/FoZcKU.png) 
    
*   Express this difference in formula:
    
    $$
    T ̂= {argmax}_T ∏_i   P(tag_i |word_i,tag _{i-1})
    $$
    
*   When estimating the single transition probability (from state q\* to state q, producing the observation o), we use the following maximum entropy model:
    
    $$
    HMM:P(Q│O)=∏_{i=1}^n   P(o_i |q_i)×∏_{i=1}^n   P(q_i |q _{i-1})  \\
    MEMM:P(Q│O)=∏_{i=1}^n   P(q_i |q _{i-1},o_i) \\
    $$
    

Decoding (Inference) of the Maximum Entropy Markov Model
--------------------------------------------------------

*   MEMM also uses the Viterbi algorithm for decoding
    
*   The general framework for decoding using the Viterbi algorithm is:
    
    $$
    P(q│q^{*},o)=\frac{1}{Z(o,q^{*})} exp⁡(\sum _i   w_i f_i (o,q))
    $$
    
*   In the HMM model, this framework is specifically realized as:
    
    $$
    v_t (j)=max_{i=1}^N⁡  v_{t-1} (i)P(s_j│s_i )P(o_t |s_j)
    $$
    
*   In MEMM, replace the likelihood and prior directly with the posterior:
    
    $$
    v_t (j)=max_{i=1}^N⁡  v_{t-1} (i) a_ij b_j (o_t)
    $$
    

Training of Maximum Entropy Markov Models
-----------------------------------------

*   MEMM as an extension of the maximum entropy model employs the same supervised algorithm for training. If there are missing label sequences in the training data, semi-supervised learning can also be performed using the EM algorithm.

Chapter 12: The Formal Grammar of English
=========================================

Compositionality
----------------

*   How are words in English composed into a phrase?
*   In other words, how do we determine that some word combinations form a part? One possibility is that these combinations can all appear in similar syntactic environments, for example, noun phrases can all appear before a verb. Another possibility comes from the prepositional and postpositional structures, for example, the prepositional phrase "on September seventeenth" can be placed at the beginning, middle, or end of a sentence, but the individual components of this phrase cannot be split and placed in different positions in the sentence, so we judge that "on September seventeenth" these three word combinations form a phrase.

Context-free grammar
--------------------

*   Context-free grammar, abbreviated as CFG, also known as phrase structure grammar, has a formalization method equivalent to the Backus-Naur form. A context-free grammar consists of two parts: rules or production, and a vocabulary.
    
*   For example, in describing noun phrases with context-free grammar, one way is that a noun phrase can be composed of a proper noun, or it can be composed of a determiner plus a nominal component, where the nominal component can consist of one or more nouns. The rules of this CFG are:
    
    *   NP→Determiner Nominal
    *   NP→ProperNoun
    *   Nominal→Noun|Noun Nominal
*   CFG can be hierarchically nested, thus the above rules can be combined with the rules below that represent lexical facts (vocabulary list):
    
    *   Det→a
    *   Det→the
    *   Noun→flight
*   Symbols are divided into two categories:
    
    *   Ultimate Symbol: A symbol corresponding to a word in reality; the lexicon is a collection of rules for introducing ultimate symbols
    *   Non 终极符号: Clustering or generalizing symbols representing ultimate symbols
*   In each rule, the right side of the arrow contains one or more terminal symbols and non-terminal symbols, and the left side of the arrow is a non-terminal symbol associated with each word and its category (part of speech).
    
*   CFG can be regarded as a mechanism for generating sentences as well as a mechanism for assigning structure to a sentence.
    
*   For example, taking the CFG mentioned earlier, an NP (noun phrase) symbol string can be generated step by step:
    
    $$
    v_t (j)=max_{i=1}^N⁡  v_{t-1} (j)P(s_j |s_i,o_t)
    $$
    
*   A flight is a derivation of NP, which is generally represented by a parse tree: ![FoZ5P1.png](https://s2.ax1x.com/2019/01/03/FoZ5P1.png) A CFG defines a formal language, which is a set of symbol strings. If a sentence derived by a grammar is within the formal language defined by that grammar, the sentence is syntactically correct. Using formal languages to simulate the grammar of natural languages is known as generative grammar.
    
*   Formal definition of context-free grammar:
    
    *   Set of non-terminating symbols (or variables)
    *   Sigma: Set of terminal symbols, disjoint from N
    *   Rule set or set of productions
    *   S: Specified start symbol
*   Some conventions defined:
    
    *   Capital letters: Represent non-terminating symbols
    *   S: Start Symbol
    *   Lowercase Greek letters: a sequence of symbols extracted from the conjunction of non-terminal and terminal symbols
    *   Lowercase Roman letters: end-of-sequence symbol string
*   Direct derivation definition: Formula to be supplemented
    
*   Exportation is a generalization of direct exportation. Subsequently, we can formally define the language L generated by the grammar G as a set of strings composed of terminal symbols, which can be exported from the specified starting symbol S through the grammar G: Formula to be supplemented
    
*   Mapping a sequence of words to its corresponding parse tree is called syntactic parsing.
    

Some grammatical rules of English
---------------------------------

*   Four most common sentence structures in English:
    *   Declarative structure: a noun phrase subject followed by a verb phrase
    *   Imperative structure: Typically begins with a verb phrase and lacks a subject
    *   Yes-no interrogative structure: commonly used for asking questions, and begins with an auxiliary verb, followed by a subject NP, and then a VP
    *   Wh interrogative structure: contains a wh phrase element
*   In the previous description, the beginning symbol was used to generate an entire sentence independently, but S can also appear on the right side of grammatical generation rules, embedded within a larger sentence. Such an S is called a clause, which has a complete semantics. Having complete semantics means that this S, in the overall syntactic parse tree of the sentence, has the main verb in its subtree with all the required arguments.

Noun phrase
-----------

*   Determiner Det: Noun phrases can begin with some simple lexical determiners, such as a, the, this, those, any, some, etc., and the position of determiners can also be replaced by more complex expressions, such as possessives. Such expressions can be recursively defined, for example, possessives plus noun phrases can constitute determiners of larger noun phrases. No determiners are needed before plural nouns or mass nouns.
*   Nominal: Comprising some modifiers before or after nouns
*   Before nouns, after determiners: Some special word classes can appear before nouns and after determiners, including cardinal numbers Card, ordinal numbers Ord, and quantity modifiers Quant.
*   Adjective Phrase AP: An adverb can appear before an adjective phrase
*   The rules for the attributive modifiers of noun phrases can be regularized as follows (items in parentheses are optional):
*   Nominal
*   Post-modifiers mainly include three types:
    *   Prepositional phrase PP: Nominal -> Nominal PP(PP)(PP)
    *   Non-restrictive relative clause: Gerund VP, Gerund VP -> GerundV NP | GerundV PP | GerundV | GerundV NP PP
    *   Relative clause: Clause starting with a relative pronoun Nominal ->Nominal RelCaluse;RelCaluse -> (who|that) VP

Consistency relationship
------------------------

*   When a verb has a noun as its subject, the phenomenon of agreement occurs; any sentence where the subject and its verb do not agree is ungrammatical, for example, the third-person singular verb without the -s ending. A set of rules can be used to expand the original grammar, making it capable of handling agreement. For example, the rule for yes-no questions is
    
    $$
    NP→Det Nominal→Det Noun→a flight
    $$
    
*   Two rules of the following form can be substituted:
    
    $$
    S \rightarrow Aux \ NP \ VP
    $$
    
*   Specify the auxiliary verb forms for the third person singular and non-third person singular separately. Such a method would lead to an increase in grammatical scale.
    

Verb phrases and subcategorization
----------------------------------

*   Verb phrases include combinations of verbs and other components, such as NP and PP, as well as combinations of both. The entire embedded sentence can also follow the verb, becoming a complement of the sentence.
*   Another potential constituent of the verb phrase is another verb phrase.
*   The verb can also be followed by a particle, which is similar to "借以," but when combined with the verb, it forms a phrasal verb that is inseparable from the verb.
*   Reclassification refers to subcategorization. Traditional grammar subcategorizes verbs into transitive verbs and intransitive verbs, while modern grammar has differentiated verbs into 100 subcategories. Discussing the relationship between verbs and possible components involves viewing verbs as predicates and components as the arguments of this predicate.
*   For the relationship between verbs and their complements, we can express the consistency features using context-free grammar and it is necessary to differentiate the various subclasses of verbs.

Auxiliary verb
--------------

*   Auxiliaries are a subclass of verbs with special syntactic constraints. Auxiliaries include modal verbs, perfect auxiliary verbs, progressive auxiliary verbs, and passive auxiliary verbs. Each auxiliary imposes a constraint on the form of the verb that follows it, and they must be combined in a certain order.
*   Four auxiliary verbs categorize the VP subcategory, with the central verbs of the VP being a bare verb, a past participle form, a present participle form, and a past participle form, respectively.
*   A sentence can use multiple auxiliary verbs, but they should be arranged in the order of modal auxiliary verbs, perfect auxiliary verbs, progressive auxiliary verbs, and passive auxiliary verbs.

Tree Diagram Database
---------------------

*   Context-free grammar can analyze a sentence into a syntactic parse tree. If all sentences in a corpus are represented in the form of a syntactic parse tree, such syntactically annotated corpus is called a treebank.
    
*   The sentences in the treebank database implicitly constitute a grammar of a language, from which we can extract CFG rules for each syntactic parse tree. The CFG rules extracted from the Penn Treebank are highly flattened, resulting in a large number of rules and long rules.
    
*   A special expression is required for searching in a treebank, which can represent constraints on nodes and connections to search for specific patterns. For example, tgrep or TGrep2.
    
*   A pattern in tgrep, TGrep2 consists of a description of a node, and a node description can be used to return a subtree rooted at this node.
    
*   One can name a class of patterns using double slashes:
    
    $$
    S \rightarrow 3sgAux \ 3sgNP \ VP \\
    S \rightarrow Non3sgAux \ Non3sgNP \ VP \\
    $$
    
*   The benefits of the Tgrep/Tgrep2 pattern lie in its ability to describe connected information. The less than symbol represents direct dominance, the much less than symbol represents dominance, and the decimal point represents linear order. This description of connections is reflected in the relationships within the parse tree as follows: ![FoZ2b4.png](https://s2.ax1x.com/2019/01/03/FoZ2b4.png) 
    

Central word and central word search
------------------------------------

*   Syntactic components can be associated with a lexical center word. In a simple lexical center word model, each context-free rule is associated with a center word, which is passed to the parse tree, so that each non-terminal symbol in the parse tree is labeled with a single word, which is the center word of that non-terminal symbol. An example is as follows: ![FoZfa9.png](https://s2.ax1x.com/2019/01/03/FoZfa9.png) 
*   To generate such a tree, each CFG rule must be expanded to recognize a right-hand constituent as the center word child node. The center word of a node is set to the center word of its child center words.
*   Another approach is to complete the search for the center word through a computational system. In this method, the search for the specified sentence is based on the context of the tree, thereby dynamically identifying the center word. Once a sentence is parsed, the tree is traversed and each node is decorated with the appropriate center word.

Grammar Equivalence and Patterns
--------------------------------

*   Grammar equivalence includes two types: strong equivalence, which means that two grammars generate the same set of symbol strings and assign the same phrase structure to each sentence; weak equivalence, which means that two grammars generate the same set of symbol strings but do not assign the same phrase structure to each sentence.
*   All grammars use a single paradigm, in which each production rule employs a specific form. For example, a context-free grammar with five senses is sigma-free, and if each production rule's form is A->BC or A->a, it indicates that this context-free grammar conforms to the Chomsky paradigm, abbreviated as CNF. All grammars in the Chomsky paradigm have a binary tree structure. Any context-free grammar can be transformed into a weakly equivalent Chomsky paradigm grammar.
*   Using a parse tree in binary tree form can produce a smaller grammar. Rules of the form A->A B are called Chomsky and-conjunctions.

Finite State Grammar and Context-Free Grammar
---------------------------------------------

*   Complex grammatical models must represent compositionality, hence they are not suitable for describing grammar using finite state models.
*   When the expansion of a non-terminal symbol also includes this non-terminal symbol, a grammatical recursion problem arises.
*   For example, using regular expressions to describe nominal-centered noun phrases: (Det)(Card)(Ord)(Quant)(AP)Nominal(PP)\*
*   To complete this regular expression, it is only necessary to expand PP in sequence, resulting in (P NP)\*. This then leads to the ground problem, because NP now appears, and NP appears in the regular expression for NP.
*   A context-free grammar can be generated by a finite automaton if and only if there exists a context-free grammar for the language L with no central self-embedding recursion.

Dependency Grammar
------------------

*   Dependency grammar contrasts with context-free grammar, where the syntactic structure is entirely described by the semantic or syntactic relationships between words and between words. An example is as follows: ![FoZOVH.png](https://s2.ax1x.com/2019/01/03/FoZOVH.png) 
*   There are no non-terminal symbols or phrase nodes; the connections in the tree only link two words. The connections, or dependency relations, represent grammatical functions or general semantic relationships, such as syntactic subjects, direct objects, indirect objects, temporal adverbials, and so on.
*   Dependency grammar has a strong predictive analytical ability and performs better in handling languages with relatively free word order.

Chapter 13: Analysis Based on Context-Free Grammar
==================================================

Analysis is searching
---------------------

*   In syntactic parsing, parsing can be seen as a search through all possible parsing tree spaces of a sentence to discover the correct parsing tree.
*   For a certain sentence (input symbol string), the goal of the parsing search is to discover all parsing trees rooted at the initial symbol S that exactly cover the entire input symbol string. The constraints on the search algorithm come from two aspects:
    *   Constraints from the data, i.e., the input sentence itself, should result in the leaves of the parsed tree being all the words of the original sentence.
    *   From grammatical constraints, the parsed tree that is searched should have a root, namely the initial symbol S
*   Top-down, goal-directed search and bottom-up, data-directed search strategies were generated based on these two constraints.
*   For top-down search, starting from the root, we continuously generate all possible child nodes at each subsequent level, searching through every possibility at each level, as shown in the figure (for the sentence "book that flight"): ![FoZh5R.png](https://s2.ax1x.com/2019/01/03/FoZh5R.png) 
*   For bottom-up parsing, the analysis starts from the input words, using the grammatical rules each time to attempt to construct an analysis tree from the bottom up. If the analysis tree successfully constructs a tree rooted at the initial symbol S and covers the entire input, then the parsing is successful. First, each word is connected to its corresponding word class through the lexicon; if a word has more than one word class, all possibilities need to be considered. Unlike top-down parsing, when moving to the next level, bottom-up parsing needs to consider whether the analyzed component matches the right-hand side of some rule, whereas top-down parsing matches the left-hand side. If a rule cannot be matched during the process, this branch is removed from the search space, as shown in the figure below: ![FoZI8x.png](https://s2.ax1x.com/2019/01/03/FoZI8x.png) 
*   Both compared:
    *   Top-down search starts from S, therefore, it will not search those subtrees that cannot be found in the tree rooted at S, while bottom-up search will generate many impossible search trees
    *   Correspondingly, top-down search is wasted on trees that cannot produce the input word sequence
    *   In summary, we need to combine top-down and bottom-up approaches

Ambiguity
---------

*   A problem that needs to be addressed in syntactic analysis is structural ambiguity, that is, grammar may yield multiple parsing results for a single sentence.
    
*   The most common two ambiguities: attachment ambiguity and coordination conjunction ambiguity.
    
*   If a particular element can attach to more than one position in the parse tree, the sentence will exhibit attachment ambiguity. For example, in the sentence "We saw the Eiffel Tower flying to Paris," "flying to Paris" can modify either the Eiffel Tower or "We."
    
*   In coordination ambiguity, there exist different phrases connected by conjunctions such as "and." For example, "old men and women" can refer to elderly men and elderly women, or elderly men and ordinary women, i.e., whether the term "old" is simultaneously assigned to both "men" and "women."
    
*   The above two ambiguities can be combined and nested to form more complex ambiguities. If we do not resolve the ambiguities but simply return all possibilities, leaving it to the user or manual judgment, the number of possibilities may increase exponentially as the analysis of sentence structure becomes more complex or as the number of analysis rules increases. Specifically, the growth of the analysis of possible sentences is similar to the arithmetic expression insertion of parentheses problem, growing exponentially according to Catalan numbers:
    
    $$
    /NNS?/    NN|NNS
    $$
    
*   Two methods exist to escape this exponential explosion:
    
    *   Dynamic programming, studying the regularity of the search space, so that common parts are derived only once, reducing the overhead related to ambiguity
    *   Employing exploratory methods to improve the search strategy of the profiler
*   Utilizing planned and backtracking search algorithms such as depth-first search or breadth-first search is a common approach in searching complex search spaces, however, the pervasive ambiguity in complex grammatical spaces makes these search algorithms inefficient, as there are many redundant search processes.
    

Dynamic Programming Analysis Method
-----------------------------------

*   In dynamic programming, we maintain a table, systematically filling in the solutions for subproblems, and using the already stored solutions for subproblems to solve larger subproblems without having to recalculate from scratch.
*   In the analysis, such a table is used to store the subtrees of various parts of the input, which are stored in the table upon discovery for later retrieval, thereby solving the problem of repeated analysis (only subtrees need to be searched without the need for re-analysis) and ambiguity issues (the analysis table implicitly stores all possible analysis results).
*   The main three dynamic programming parsing methods are the CKY algorithm, the Earley algorithm, and the table parsing algorithm.

### CKY Parsing

*   CKY parsing requires that the grammar must satisfy the Chomsky paradigm, i.e., the right-hand side of a generating rule must either be two non-terminals or one terminal symbol. If it is not in Chomsky 范式, then a general CFG needs to be transformed into CNF:
    *   Right-hand side has both terminal symbols and non-terminal symbols: Create a separate non-terminal symbol for the right-hand terminal symbol, for example: INF-VP → to VP, change to INF-VP → TO VP and TO → to
    *   There is only one non-terminal symbol on the right: This non-terminal symbol is called a unit product, which will eventually generate non-unit products, and the unit products are replaced by the rules of the non-unit products generated in the end
    *   Right side has more than 2 symbols: introducing new non-terminal symbols to decompose the rules
    *   Lexical rules remain unchanged, but new lexical rules may be generated during the transformation process
*   After all the rules are converted to CNF, the non-terminal symbols in the table have two child nodes during parsing, and each entry in the table represents an interval in the input. For example, for an entry such as \[0,3\], it can be split into two parts, where one part is \[0,2\], and the other part is \[2,3\]. The former is on the left side of \[0,3\], and the latter is directly below \[0,3\], as shown in the following figure: ![FoZo26.png](https://s2.ax1x.com/2019/01/03/FoZo26.png) 
*   The next step is how to fill out the table, and we analyze it through a bottom-up approach. For each entry \[i, j\], the table cells within the input interval from i to j contribute to this entry value, that is, the cells to the left and below the entry \[i, j\]. The CKY pseudo-algorithm diagram in the following table describes this process: ![FoZjIA.png](https://s2.ax1x.com/2019/01/03/FoZjIA.png) 
*   Outer loop iterates over columns from left to right, inner loop iterates over rows from bottom to top, and the innermost loop traverses all possible binary substrings of \[i, j\]. The table stores the set of non-terminal symbols that can represent the symbol string in the interval \[i, j\]. Since it is a set, there will be no repeated non-terminal symbols.
*   Now that we have completed the recognition task, the next step is to analyze. Analysis involves finding a non-terminal symbol as the starting symbol S that corresponds to the entire sentence within the interval \[0, N\]. Firstly, we need to make two modifications to the algorithm:
    *   The stored information in the table is not only non-terminal symbols but also their corresponding pointers, which point to the table entry that generates the non-terminal symbol
    *   Permit different versions of the same non-terminal symbol to exist within an entry
*   After making these changes, the table contains all possible parsing information for a given input. We can choose any non-terminal symbol from the \[0,N\] entries as the starting symbol S, and then iteratively extract the parsing information according to the pointer.
*   Of course, returning all possible decompositions would encounter an exponential explosion problem, therefore, we apply the Viterbi algorithm to the complete table, calculate the decomposition with the highest probability, and return this decomposition result.

### Early Algorithms

*   Compared to CKY's bottom-up parsing, the Early algorithm employs a top-down parsing approach and uses a one-dimensional table to save the state, with each state containing three types of information:
    *   Corresponding subtree for a single grammatical rule
    *   Completion state of subtrees
    *   Subtrees correspond to positions in the input
*   Algorithm flowchart as follows: ![FoZHKO.png](https://s2.ax1x.com/2019/01/03/FoZHKO.png) 
*   Algorithms have three operations on states:
    *   Prediction: Create a new state to represent the top-down prediction generated during the analysis process. When the state to be analyzed is neither a terminal symbol nor a category of lexical types, a new state is created for each different expansion of this non-terminal symbol.
    *   Scan: When the state to be analyzed is a lexical category, check the input symbol string and add the state corresponding to the predicted lexical category to the syntax diagram.
    *   Completion: When all the state analyses on the right are completed, the completion operation searches for the grammatical category at this position in the input, discovers, and advances all the states created earlier.

### Table analysis

*   Table decomposition allows for the dynamic determination of the order of table processing, where the algorithm dynamically removes an edge from the graph according to a plan, and the order of elements in the plan is determined by rules. ![FoZTxK.png](https://s2.ax1x.com/2019/01/03/FoZTxK.png) 

------------

*   Sometimes we only need to input partial syntactic analysis information of a sentence
*   The task of partial analysis can be accomplished by cascading finite state automata, which will produce a more "flat" analysis tree than the previously mentioned methods.
*   Another effective method of partial parsing is segmentation. Using the most widely covered grammar for part-of-speech tagging of sentences, dividing them into sub-blocks with main part-of-speech tagging information and no recursive structure, where the sub-blocks do not overlap, is segmentation.
*   We enclose each block with square brackets, and some words may not be enclosed, belonging to the blocks outside.
*   The most important aspect of partitioning is that the basic partitioning cannot recursively contain the same type of components.

### Rule-based Finite State Partitioning

*   Utilizing a finite state method for segmentation requires manually constructing rules for specific purposes, then finding the longest matching segment from left to right, and proceeding to segment in order from there. This is a greedy segmentation process that does not guarantee the globally optimal solution.
*   The main limitation of these partitioning rules is that they cannot contain recursion.
*   The advantage of using finite state segmentation lies in the ability to utilize the output of the previous transducer as input to form a cascade, in partial parsing, this method can effectively approximate the true context-free parser.

### Block-based Machine Learning

*   Chunking can be regarded as a sequence classification task, where each position is classified as 1 (chunk) or 0 (not chunked). Machine learning methods used for training sequence classifiers can all be applied to chunking.
*   A highly effective method is to treat segmentation as a sequence labeling task similar to part-of-speech tagging, encoding both segmentation information and the labeling information of each block with a small set of labeling symbols. This method is called IOB labeling, with B representing the beginning of a block, I indicating within a block, and O indicating outside a block. Both B and I are followed by suffixes, representing the syntactic information of the block.
*   Machine learning requires training data, and it is difficult to obtain segmented labeled data. One method is to use existing treebank resources, such as the Penn Treebank.

### Evaluation Block System

*   Accuracy: The number of correctly segmented blocks given by the model / The total number of blocks given by the model
*   Recall rate: Number of correctly segmented blocks given by the model / Total number of correctly segmented blocks in the text
*   F1 score: Harmonic mean of accuracy and recall

Chapter 14: Statistical Analysis
================================

Probabilistic Context-Free Grammar
----------------------------------

*   Probability Context-Free Grammar (PCFG) is a simple extension of context-free grammar, also known as stochastic context-free grammar. PCFG makes a slight change in definition:
    *   N: Non-terminal symbol set
    *   Σ: Set of termination symbols
    *   R: Rule set, similar to the context-free grammar, but with an additional probability p, representing the conditional probability of executing a particular rule $C(n)=\frac{1}{1+n} C_{2n}^n$
    *   A specified starting symbol
*   When the sum of the probabilities of all sentences in a language is 1, we say that the PCFG is consistent. Some recursive rules can lead to an inconsistent PCFG.

PCFG for Disambiguation
-----------------------

*   The probability of a specific parsing for a given sentence is the product of all rule probabilities, which is both the probability of the parsing and the joint probability of the parsing and the sentence. Thus, for sentences with parsing ambiguities, the probabilities of different parses are different, and ambiguity can be resolved by choosing the parse with a higher probability.

PCFG for Language Modeling
--------------------------

*   PCFG assigns a probability to a sentence (i.e., the probability of parsing), thus it can be used for language modeling. Compared to n-gram grammar models, PCFG considers the entire sentence when calculating the conditional probability of generating each word, resulting in better performance. For ambiguous sentences, the probability is the sum of the probabilities of all possible parses.

Probability CKY Parsing of PCFG
-------------------------------

*   PCFG probabilistic parsing problem: generating the most probable parsing for a sentence
*   The probability CKY algorithm extends the CKY algorithm, encoding each part of the CKY parsing tree into a matrix of $P(\beta|A)$ . Each element of the matrix contains a probability distribution over a set of non-terminal symbols, which can be considered as each element also being V-dimensional, thus the entire storage space is $(n+1)*(n+1)$ . Where \[i,j,A\] represents the probability that the non-terminal symbol A can be used to represent the segment from position i to j in the sentence.
*   Algorithm Pseudocode: ![FoZbrD.png](https://s2.ax1x.com/2019/01/03/FoZbrD.png) 
*   It can be seen that the method also divides the interval \[i, j\] using k, takes the combination with the highest probability as the probability of the interval, and then expands the interval to the right for dynamic programming.

Learning the rule probabilities of PCFG
---------------------------------------

*   The pseudo-algorithm diagram above uses the probability of each rule. How to obtain this probability? There are two methods, the first being a naive approach: using classical probability type statistics on a known treebank dataset:
    
    $$
    (n+1)*(n+1)*V
    $$
    
*   If we do not have a treebank, we can use a non-probabilistic parsing algorithm to parse a dataset and then calculate the probabilities. However, non-probabilistic parsing algorithms require calculating probabilities for each possible parsing when parsing ambiguous sentences, but calculating probabilities requires a probabilistic parsing algorithm, thus falling into a chicken-and-egg cycle. One solution is to first use a uniform probability parsing algorithm to parse the sentence, obtain the probability of each parsing, then use probability-weighted statistics, and then re-estimate the probability of parsing rules, continue parsing, and iteratively repeat until convergence. This algorithm is called the inside-outside algorithm, which is an extension of the forward-backward algorithm and is also a special case of the EM algorithm.
    

PCFG issue
----------

*   The independence assumption leads to poor modeling of the structural dependency of the parse tree: each PCFG rule is assumed to be independent of other rules, for example, statistical results indicate that pronouns are more likely to be subjects than nouns, so when an NP is expanded, if the NP is a subject, it is more likely to be expanded into a pronoun—here, the position of NP in the sentence needs to be considered, however, this probabilistic dependency relationship is not allowed by PCFG
*   Lack of sensitivity to specific words leads to issues such as subcategorization ambiguity, preposition attachment ambiguity, and coordination structure ambiguity: for example, in the preposition attachment issue, which part does a prepositional phrase like "into Afghanistan" attach to, and in the calculation of PCFG, it is abstracted as which part a prepositional phrase should attach to, while the abstraction probability comes from the statistics of the corpus, which does not consider specific words. For example, in the coordination structure ambiguity, if two possible parse trees of a sentence use the same rules but the rules are located differently in the trees, PCFG calculates the same probability for the two parses: because PCFG assumes that rules are independent, the joint probability is the product of individual probabilities.

Improving PCFG by Splitting and Merging Non-Terminals
-----------------------------------------------------

*   Address the issue of structural dependency first. It was previously mentioned that we hope for different probability rules for NP as a subject and object. One idea is to split NP into subject NP and object NP. The method to achieve this split is parent node annotation, where each node is annotated with its parent node. For subject NP, the parent node is S, and for object NP, the parent node is VP, thus distinguishing different NPs. In addition, the parsing tree can be enhanced by splitting words according to their parts of speech.
*   Splitting leads to an increase in rules, with less data available to train each rule, causing overfitting. Therefore, a handwritten rule or an automatic algorithm should be used to merge some splits based on each training set.

Probabilistic lexicalization CFG
--------------------------------

*   Probability CKY parsing modifies the grammatical rules, while the probabilistic lexicalization model modifies the probability model itself. For each rule, not only should the rule changes for the constituents be generated, but also the headword and part of speech of each constituent should be annotated, as shown in the following figure: ![FoeSRP.png](https://s2.ax1x.com/2019/01/03/FoeSRP.png) 
    
*   To generate such an analysis tree, each rule on the right side of a PCFG needs to select a constituent as a central word sub-node, using the central word and part of speech of the sub-node as the central word and part of speech of the node. Among them, the rules are divided into two categories, internal rules and lexical rules; the latter is deterministic, while the former requires us to estimate: ![FoZqqe.png](https://s2.ax1x.com/2019/01/03/FoZqqe.png) 
    
*   We can split the rules using the idea of similar parent node annotation, with each part corresponding to a possible choice of a central word. If we treat the CFG with probabilistic vocabulary as a large CFG with many rules, we can estimate the probability using the previous classical probability model. However, such an effect will not be very good, because the rule division is too fine, and there is not enough data to estimate the probability. Therefore, we need to make some independence assumptions, decomposing the probability into smaller probability products, which can be easily estimated from the corpus.
    
*   Different statistical analyzers differ in the independence assumptions they make.
    
*   Collins' analysis is shown as follows: ![FoZzGt.png](https://s2.ax1x.com/2019/01/03/FoZzGt.png) 
    
*   The probability is decomposed as follows:
    
    $$
    P(\alpha \rightarrow \beta | \alpha) = \frac{Count(\alpha \rightarrow \beta)}{\sum _{\gamma} Count(\alpha \rightarrow \gamma)}
    $$
    
*   After generating the left side of the generative expression, the central word of the rule is first generated, followed by the generation of the dependency of the central word one by one from the inside out. Starting from the left side of the central word, generation continues until the STOP symbol is encountered, after which the right side is generated. After making a probability split as shown in the above expression, each probability is easy to statistically calculate from a smaller amount of data. The complete Collins parser is more complex, taking into account word distance relationships, smoothing techniques, unknown words, and so on.
    

Evaluation Analyzer
-------------------

*   PARSEVAL measure is the standard method for analyzer evaluation, for each sentence s:
    *   Recall rate = (Count of correct components in s's candidate parsing) / (Count of correct components in s's treebank)
    *   Accuracy of tagging = (Count of correct components in candidate parsing of s) / (Count of all components in candidate parsing of s)

Discriminant reordering
-----------------------

*   PCFG parsing and Collins lexical parsing both belong to generative parsers. The drawback of generative models is that it is difficult to introduce arbitrary information, i.e., it is difficult to incorporate features that are locally irrelevant to a particular PCFG rule. For example, the feature that parsing trees tend to be right-generated is not convenient to add to the generative model.
*   There are two types of discriminative models for syntactic parsing, those based on dynamic programming and those based on discriminative reordering.
*   Discriminant reordering consists of two stages. In the first stage, we use a general statistical profiler to generate the top N most probable profiles and their corresponding probability sequences. In the second stage, we introduce a classifier, which takes a series of sentences and the top N profile-probability pairs of each sentence as input, extracts a large set of features, and selects the best profile for each sentence. Features include: profile probability, CFG rules in the profile tree, the number of parallel and coordinate structures, the size of each component, the extent of right generation in the tree, the binary grammar of adjacent non-terminal symbols, the frequency of different parts of the tree, and so on.

Language Modeling Based on Analysis
-----------------------------------

*   The simplest way to use a statistical parser for language modeling is to employ the two-phase algorithm previously mentioned. In the first phase, we run a standard speech recognition decoder or machine translation decoder (based on ordinary N-gram grammar) to generate N best candidate sentences; in the second phase, we run the statistical parser and assign a probability to each candidate sentence, selecting the one with the highest probability.

Human anatomy
-------------

*   Humans also employ similar probabilistic parsing ideas in recognizing sentences; two examples:
    *   For frequently occurring binary grammatical structures, people spend less time reading these binary grammatical structures
    *   Some experiments indicate that humans tend to choose the analysis with a higher statistical probability during disambiguation


{% endlang_content %}

{% lang_content zh %}

# 第二章：正则表达式与自动机

- 正则表达式：一种用于查找符合特定模式的子串或者用于以标准形式定义语言的工具，本章主要讨论其用于查找子串的功能。正则表达式用代数的形式来表示一些字符串集合。
- 正则表达式接收一个模式，然后在整个语料中查找符合这个模式的子串，这个功能可以通过设计有限状态自动机实现。
- 字符串看成符号的序列，所有的字符，数字，空格，制表符，标点和空格均看成符号。

## 基本正则表达式模式

- 用双斜线表示正则表达式开始和结束（perl中的形式）
  - 查找子串，大小写敏感：/woodchuck/-> woodchuck
  - 用方括号代表取其中一个，或：/[Ww]oodchuck/->woodchuck or Woodchuck
  - 方括号加减号，范围内取或：/[2-5]/->/[2345]
  - 插入符号放在左方括号后，代表模式中不出现后接的所有符号，取非: /^Ss/ ->既不是大写S也不是小写s
  - 问号代表之前的符号出现一个或不出现：/colou?r/->color or colour
  - 星号代表之前的符号出现多个或不出现：/ba*/->b or ba or baa or baaa......
  - 加号代表之前的符号出现至少一次：/ba+/->ba or baa or baaa.......
  - 小数点代表通配符，与任何除了回车符之外的符号匹配：/beg.n/->begin or begun or beg’n or .......
  - 锚符号，用来表示特定位置的子串，插入符号代表行首，美元符号代表行尾，\b代表单词分界线，\B代表单词非分界线，perl将单词的定义为数字、下划线、字母的序列，不在其中的符号便可以作为单词的分界。

## 析取、组合和优先

- 用竖线代表析取，字符串之间的或：/cat|dog/->cat or dog
- 用圆括号代表部分析取（组合），圆括号内也可以用基本算符：/gupp(y|ies)/->guppy or guppies
- 优先级：圆括号>计数符>序列与锚>析取符

## 高级算符

- \d：任何数字
- \D：任何非数字字符
- \w：任何字母、数字、空格
- \W：与\w相反
- \s：空白区域
- \S：与\s相反
- {n}：前面的模式出现n个
- {n,m}：前面的模式出现n到m个
- {n,}：前面的模式至少出现n个
- \n：换行
- \t：表格符

## 替换、寄存器

- 替换s/A/B/：A替换成B
- s/(A)/<\1>/：用数字算符\1指代A，在A的两边加上尖括号
- 在查找中也可以用数字算符，指代圆括号内内容，可以多个算符指代多个圆括号内内容
- 这里数字算符起到了寄存器的作用

## 有限状态自动机

- 有限状态自动机和正则表达式彼此对称，正则表达式是刻画正则语言的一种方法。正则表达式、正则语法和自动状态机都是表达正则语言的形式。FSA用有向图表示，圆圈或点代表状态，箭头或者弧代表状态转移，用双圈表示最终状态，如下图表示识别/baa+!/的状态机图： 
  ![FoVj3V.png](https://s2.ax1x.com/2019/01/03/FoVj3V.png)
- 状态机从初始状态出发，依次读入符号，若满足条件，则进行状态转移，若读入的符号序列满足模式，则状态机可以到达最终状态；若符号序列不满足模式，或者自动机在某个非最终状态卡住，则称自动机拒绝了此次输入。
- 另一种表示方式是状态转移表：
  ![FoVqNn.png](https://s2.ax1x.com/2019/01/03/FoVqNn.png)
- 一个有限自动机可以用5个参数定义：
  - $Q$：状态{q_i}的有限集合
  - \sum ：有限的输入符号字母表
  - $q_0$：初始状态
  - $F$：终极状态集合
  - $\delta (q,i)$：状态之间的转移函数或者转移矩阵，是从$Q × \Sigma$到$2^Q$的一个关系
- 以上描述的自动机是确定性的，即DFSA，在已知的记录在状态转移表上的状态时，根据查表自动机总能知道如何进行状态转移。算法如下，给定输入和自动机模型，算法确定输入是否被状态机接受：
  ![FoZpB4.png](https://s2.ax1x.com/2019/01/03/FoZpB4.png)
- 当出现了表中没有的状态时自动机就会出错，可以添加一个失败状态处理这些情况。

## 形式语言

- 形式语言是一个模型，能且只能生成和识别一些满足形式语言定义的某一语言的符号串。形式语言是一种特殊的正则语言。通常使用形式语言来模拟自然语言的某些部分。以上例/baa+!/为例，设对应的自动机模型为m，输入符号表$\Sigma = {a,b,!}$，$L(m)$代表由m刻画的形式语言，是一个无限集合${baa!,baaa!,baaaa!,…}$

## 非确定有限自动机

- 非确定的有限自动机NFSA,把之前的例子稍微改动，自返圈移动到状态2，就形成了NFSA，因为此时在状态2，输入a，有两种转移可选，自动机无法确定转移路径：
  ![FoVLhq.png](https://s2.ax1x.com/2019/01/03/FoVLhq.png)
- 另一种NFSA的形式是引入$\epsilon$转移，即不需要输入符号也可以通过此$\epsilon$转移进行转移，如下图，在状态3时依然不确定如何进行转移：
  ![FoVX90.png](https://s2.ax1x.com/2019/01/03/FoVX90.png)
- 在NFSA时，面临转移选择时自动机可能做出错误的选择，此时存在三种解决方法：
  - 回退：标记此时状态，当确定发生错误选择之后，回退到此状态
  - 前瞻：在输入中向前看，帮助判定进行选择
  - 并行：并行的进行所有可能的转移
- 在自动机中，采用回退算法时需要标记的状态称为搜索状态，包括两部分：状态节点和输入位置。对于NFSA，其状态转移表也有相应改变，如图，添加了代表$\epsilon$转移的$\epsilon$列，且转移可以转移到多个状态：
  ![FoZE36.png](https://s2.ax1x.com/2019/01/03/FoZE36.png)
- 采用回退策略的非确定自动机算法如下，是一种搜索算法： 
  ![FoZSuF.png](https://s2.ax1x.com/2019/01/03/FoZSuF.png)
- 子函数GENERATE-NEW-STATES接受一个搜索状态，提取出状态节点和输入位置，查找这个状态节点上的所有状态转移可能，生成一个搜索状态列表作为返回值；
- 子函数ACCEPT-STATE接受一个搜索状态，判断是否接受，接受时的搜索状态应该是最终状态和输入结束位置的二元组。
- 算法使用进程表（agenda）记录所有的搜索状态，初始只包括初始的搜索状态，即自动机初始状态节点和输入起始。之后不断循环，从进程表中调出搜索状态，先调用ACCEPT-STATE判断是否搜索成功，之后再调用GENERATE-NEW-STATES生成新的搜索状态加入进程表。循环直到搜索成功或者进程表为空（所有可能转移均尝试且未成功）返回拒绝。
- 可以注意到NFSA算法就是一种状态空间搜索，可以通过改变搜索状态的顺序提升搜索效率，例如用栈实现进程表，进行深度优先搜索DFS；或者使用队列实现进程表，进行宽度优先搜索BFS。
- 对于任何NFSA，存在一个完全等价的DFSA。

## 正则语言和NFSA

- 定义字母表\sum 为所有输入符号集合；空符号串$\epsilon$，空符号串不包含再字母表中；空集∅。在\sum 上的正则语言的类（或者正则集）可以形式的定义如下：
  - ∅是正则语言
  - ∀a ∈ $\sum$ ∪$\epsilon$,{a}是形式语言
  - 如果$L_1$和$L_2$是正则语言，那么：
  - $L_1$和$L_2$的拼接是正则语言
  - $L_1$和$L_2$的合取、析取也是正则语言
  - $L_1$^*，即$L_1$的Kleene闭包也是正则语言
- 可见正则语言的三种基本算符：拼接、合取及析取、Kleene闭包。任何正则表达式可以写成只使用这三种基本算符的形式。
- 正则语言对以下运算也封闭（$L_1$和$L_2$均为正则语言）：
  - 交：$L_1$和$L_2$的符号串集合的交构成的语言也是正则语言
  - 差：$L_1$和$L_2$的符号串集合的差构成的语言也是正则语言
  - 补：不在$L_1$的符号串集合中的集合构成的语言也是正则语言
  - 逆：$L_1$所有符号串的逆构成的集合构成的语言也是正则语言
- 可以证明正则表达式和自动机等价，一个证明任何正则表达式可以建立对应的自动机的方法是，根据正则语言的定义，构造基础自动机代表$\epsilon$、∅以及$\sum$中的单个符号a，然后将三种基本算符表示为自动机上的操作，归纳性的，在基础自动机上应用这些操作，得到新的基础自动机，这样就可以构造满足任何正则表达式的自动机，如下图：
  ![FoVxjU.png](https://s2.ax1x.com/2019/01/03/FoVxjU.png)
  基础自动机
  ![FoZPE9.png](https://s2.ax1x.com/2019/01/03/FoZPE9.png) 
  拼接算符
  ![FoZ9HJ.png](https://s2.ax1x.com/2019/01/03/FoZ9HJ.png)
  Kleene闭包算符
  ![FoZiNR.png](https://s2.ax1x.com/2019/01/03/FoZiNR.png)
  合取析取算符

# 第三章：形态学与有限状态转录机

- 剖析：取一个输入并产生关于这个输入的各类结构

## 英语形态学概论

- 形态学研究词的构成，词可以进一步拆解为语素，语素可分为词干和词缀，词缀可分为前缀、中缀、后缀、位缀。
- 屈折形态学：英语中，名词只包括两种屈折变化：一个词缀表示复数，一个词缀表示领属：
  - 复数：-s，-es，不规则复数形式
  - 领属：-‘s，-s’
- 动词的屈折变化包括规则动词和非规则动词的变化：
  - 规则动词：主要动词和基础动词，-s，-ing，-ed，
  - 非规则动词
- 派生形态学：派生将词干和一个语法语素结合起来，形成新的单词
  - 名词化：-ation，-ee，-er，-ness
  - 派生出形容词：-al，-able，-less

## 形态剖析

- 例子：我们希望建立一个形态剖析器，输入单词，输出其词干和有关的形态特征，如下表，我们的目标是产生第二列和第四列：
  ![FoZA9x.png](https://s2.ax1x.com/2019/01/03/FoZA9x.png)
- 我们至少需要：
  - 词表（lexicon）：词干和词缀表及其基本信息
  - 形态顺序规则（morphotactics）：什么样的语素跟在什么样的语素之后
  - 正词法规则（orthographic rule）：语素结合时拼写规则的变化
- 一般不直接构造词表，而是根据形态顺序规则，设计FSA对词干进行屈折变化生成词语。例如一个名词复数化的简单自动机如下图：
  ![FoZmuD.png](https://s2.ax1x.com/2019/01/03/FoZmuD.png)
- 其中reg-noun代表规则名词，可以通过加s形成复数形式，并且忽略了非规则单数名词(irreg-sg-noun)和非规则复数名词(irreg-pl-noun)。另外一个模拟动词屈折变化的自动机如下图：
  ![FoZQUA.png](https://s2.ax1x.com/2019/01/03/FoZQUA.png)
- 使用FSA解决形态识别问题（判断输入符号串是否合法）的一种方法是，将状态转移细分到字母层次，但是这样仍然会存在一些问题：
  ![FoZZjO.png](https://s2.ax1x.com/2019/01/03/FoZZjO.png)

## 有限状态转录机

- 双层形态学：将一个词表示为词汇层和表层，词汇层表示该词语素之间的简单毗连（拼接，concatenation），表层表示单词实际最终的拼写，有限状态转录机是一种有限状态自动机，但其实现的是转录，实现词汇层和表层之间的对应，它有两个输入，产生和识别字符串对，每一个状态转移的弧上有两个标签，代表两个输入。
  ![FoZVgK.png](https://s2.ax1x.com/2019/01/03/FoZVgK.png)
- 从四个途径看待FST：
  - 作为识别器：FST接受一对字符串，作为输入，如果这对字符串在语言的字符串对中则输出接受否则拒绝
  - 作为生成器：生成语言的字符串对
  - 作为翻译器：读入一个字符串，输出另一个
  - 作为关联器：计算两个集合之间的关系
- 定义有限状态转录机：
  - Q：状态{q_i}的有限集合
  - \sum ：有限的输入符号字母表
  - ∆：有限的输出符号字母表
  - $q_0 \in Q$：初始状态
  - $F⊆Q$：终极状态集合
  - $\delta (q,w)$：状态之间的转移函数或者转移矩阵，是从Q×\sum 到2^Q的一个关系，q是状态，w是字符串，返回新状态集合
  - $\sigma (q,w)$：输出函数，给定每一个状态和输入，返回可能输出字符串的集合，是从$Q × \Sigma$到$2^∆$的一个关系
- 在FST中，字母表的元素不是单个符号，而是符号对，称为可行偶对。类比于FSA和正则语言，FST和正则关系同构，对于并运算封闭，一般对于差、补、交运算不封闭。
- 此外，FST，
  - 关于逆反（逆的逆）闭包，逆反用于方便的实现作为剖析器的FST到作为生成器的FST的转换
  - 关于组合（嵌套）闭包，用于将多个转录机用一个更复杂的转录机替换。
- 转录机一般是非确定性的，如果用FSA的搜索算法会很慢，如果用非确定性到确定性的转换算法，则有些FST本身是不可以被转换为为确定的。
- 顺序转录机是一种输入确定的转录机，每个状态转移在给定状态和输入之后是确定的，不像上图中的FST，状态0在输入b时有两种状态转移（转移到相同的状态，但是输出不同）。顺序转录机可以使用$\epsilon$符号，但是只能加在输出字符串上，不能加在输入字符串上，如下图： 
  ![FoZuHH.png](https://s2.ax1x.com/2019/01/03/FoZuHH.png)
- 顺序转录机输出不一定是序列的，即从同一状态发出的不同转移可能产生相同输出，因此顺序转录机的逆不一定是顺序转录机，所以在定义顺序转录机时需要定义方向，且转移函数和输出函数需要稍微修改，输出空间缩小为Q和∆。
- 顺序转录机的一种泛化形式是并发转录机，其在最终状态额外输出一个字符串，拼接到已经输出的字符串之后。顺序和并发转录机的效率高，且有有效的算法对其进行确定化和最小化，因此很重要。P并发转录机在此基础上可以解决歧义问题。

## 用有限状态转录机进行形态剖析

- 将单词看成词汇层和表层之间的关系，如下图： 
  ![FoZnDe.png](https://s2.ax1x.com/2019/01/03/FoZnDe.png)
- 在之前双层形态学的基础定义上，定义自己到自己的映射为基本对，用一个字母表示；用^代表语素边界；用#代表单词边界，在任务中提到需要输出+SG之类的语素特征，这些特征在另一个输出上没有对应的输出符号，因此映射到空字符串或边界符号。我们把输入输出对用冒号连接，也可以写在弧的上下。一个抽象的表示英语名词复数屈折变化的转录机如下图：
  ![FoZl4I.png](https://s2.ax1x.com/2019/01/03/FoZl4I.png)
- 之后我们需要更新词表，使得非规则复数名词能够被剖析为正确的词干：
  ![FoZMEd.png](https://s2.ax1x.com/2019/01/03/FoZMEd.png)
- 之后将抽象的转录机写成具体的，由字母组成转移弧的转录机，如下图，只展示了具体化部分非规则复数和单数名词之后的转录机：
  ![FoZ3Ct.png](https://s2.ax1x.com/2019/01/03/FoZ3Ct.png)

## 转录机和正词法规则

- 用拼写规则，也就是正词法规则来处理英语中经常在语素边界发生拼写错误的问题。
- 以下是一些拼写规则实例：
  - 辅音重叠：beg/beggin
  - E的删除：make/making
  - E的插入：watch/watches
  - Y的替换：try/tries
  - K的插入：panic/panicked
- 为了实现拼写规则，我们在词汇层和表层之间加入中间层，以符合特定规则的语素毗连作为输入，以修改之后的正确的语素毗连作为输出，例如fox +N +PL输入到中间层即第一次转录，得到fox ^ s #，之后中间层到表层的第二次转录检测到特殊语素毗连：x^和s#，就在表层的x和s之间插入一个e，得到foxes。下面的转录机示意图展示了这个过程：
  ![FoZ88P.png](https://s2.ax1x.com/2019/01/03/FoZ88P.png)
- 这个转录机只考虑x^和s#毗连需插入e这一正词法规则
- 其他的词能正常通过
- $Q_0$代表无关词通过，是接受状态
- $Q_1$代表看见了zsx，作为中间状态保存，一直保存的是最后的与语素毗连的z,s,x，如果出现了其他字母则返回到q0，其本身也可以作为接受态
- $Q_2$代表看见了与z,s,x毗连的语素，这之后有四种转移
  - 接了$x$,$z$，回到$q_1$，也就是认为重新接到了可能和语素毗连的x,z
  - 接了$s$，分为两种情况，一种是正常需要插入e，这时通过$\epsilon$转移到$q_3$再到$q_4$；另一种是本来就需要插入$e$，这就到达$q_5$，之后视情况回退了$q_1$、$q_0$，或者$s$又毗连语素回到$q_2$。两种情况不确定，需要通过搜索解决
  - 接单词边界和其他符号，回到$q_0$
  - $q_2$本身也可以作为接受态

## 结合

- 现在可以通过三层结构，结合产生语素和进行正词法规则矫正的转录机。从词汇层到中间层用一个转录机产生语素，从中间层到表层可并行使用多个转录机进行正词法规则的矫正。
- 两类转录机叠加时可以改写成一类转录机，这时需要对两类状态机状态集合计算笛卡尔积，对新集合内每一个元素建立状态。
- 这种三层结构是可逆的，但是进行剖析时（从表层到词汇层）会出现歧义问题，即一个单词可能剖析出多种语素结合，这时单纯依靠转录机无法消歧，需要借助上下文。

## 其他应用（简单介绍）

- 不需要词表的FST，PORTER词干处理器：将层叠式重写规则用FST实现，提取出单词的词干。
- 分词和分句：一个简单的英文分词可以基于正则表达式实现，一个简单的中文分词可以通过maxmatch（一种基于最大长度匹配的贪婪搜索算法）实现。
- 拼写检查与矫正：使用了投影操作的FST可以完成非词错误的检测，然后基于最小编辑距离（使用动态规划算法实现）可以矫正。正常词错误检测和矫正需借助N元语法模型。

## 人如何进行形态处理

- 研究表明，人的心理词表存储了一部分形态机构，其他的结构不组合在心理词表中，而需要分别提取并组合。研究说明了两个问题：
  - 形态尤其是屈折变化之类的能产性形态在人的心理词表中起作用，且人的语音词表和正词法词表可能具有相同结构。
  - 例如形态这种语言处理的很多性质，可以应用于语言的理解和生成。

# 第四章：N元语法

- 语言模型是关于单词序列的统计模型，N元语法模型是其中的一种，它根据之前N-1个单词推测第N个单词，且这样的条件概率可以组成整个单词序列（句子）的联合概率。

## 在语料库中统计单词

- 区别：word type或者叫 vocabulary size V，代表语料中不同单词的个数，而tokens，不去重，代表语料的大小。有研究认为词典大小不低于tokens数目的平方根。
  非平滑N元语法模型
- 任务：根据以前的单词推断下一个单词的概率：$P(w|h)$，以及计算整个句子的概率$P(W)$。
- 最朴素的做法是用古典概型，统计所有历史h和当前词w组成的片段在整个语料中出现的次数，并除以历史h片段在整个语料中出现的次数。句子的概率也用相似的方法产生。缺点：依赖大语料，且语言本身多变，这样的计算限制过于严格。
- 接下来引入N元语法模型，首先通过概率的链式法则，可以得到条件概率$P(w|h)$和整个句子的联合概率$P(W)$之间的关系：
  
  $$
  P(w_1^n) = P(w_1)P(w_2|w_1)P(w_3|w_1^2)...P(w_n|w_1^{n-1}) \\
= \prod _{k=1}^n P(w_k|w_1^{k-1}) \\
  $$
- N元语法模型放松了条件概率的限制，做出一个马尔可夫假设：每个单词的概率只和它之前N-1个单词相关，例如二元语法模型，只和前一个单词相关，用这个条件概率去近似$P(w|h)$:
  
  $$
  P(w_n|w_1^{n-1}) \approx P(w_n|w_{n-1}) \\
  $$
- N元语法模型里的条件概率用最大似然估计来估算，统计语料中各种N元语法的个数，并归一化，其中可以简化的一点是：以二元语法为例，所有给定单词开头的二元语法总数必定等于该单词一元语法的计数：
  
  $$
  P(w_n|w_{n-1}) = \frac {C(w_{n-1}w_n)}{C(w_{n-1})} \\
  $$
- 使用N元语法之后，句子概率的链式分解变得容易计算，我们可以通过计算各种句子的概率来判断句子是否包含错字，或者计算某些句子在给定上下文中出现的可能，因为N元语法能捕捉一些语言学上的特征，或者一些用语习惯。在语料充足的时候，我们可以使用三元语法模型获得更好的效果。

## 训练集和测试集

- N元语法模型对训练集非常敏感。N元语法的N越大，依赖的上下文信息越多，利用N元语法模型生成的句子就越流畅，但这些未必“过于流畅”，其原因在于N元语法概率矩阵非常大且非常稀疏，在N较大例如四元语法中，一旦生成了第一个单词，之后可供的选择非常少，接着生成第二个单词之后选择更少了，往往只有一个选择，这样生成的就和原文中某一个四元语法一模一样。过于依赖训练集会使得模型的泛化能力变差。因此我们选择的训练集和测试集应来自同一细分领域。
- 有时候测试集中会出现训练集词典里没有的词，即出现未登录词（Out Of Vocabulty,OOV）。在开放词典系统中，我们先固定词典大小，并将所有未登录词用特殊符号<UNK>代替，然后才进行训练。

## 评价N元语法模型：困惑度

- 模型的评价分两种：外在评价和内在评价。外在评价是一种端到端的评价，看看某一模块的改进是否改进了整个模型的效果。内在评价的目的是快速衡量模块的潜在改进效果。内在评价的潜在改进效果不一定会使得端到端的外在评价提高，但是一般两者都存在某种正相关关系。
- 困惑度（Perplexsity,PP）是一种关于概率模型的内在评价方法。语言模型的在测试集上的困惑度是语言模型给测试集分配的概率的函数。以二元语法为例，测试集上的困惑度为：
  
  $$
  PP(W) = \sqrt[n]{\prod _{i=1}^N \frac {1}{P(w_i|w_{i-1})}} \\
  $$
- 概率越高，困惑度越低。困惑度的两种解释：
  - 加权的平均分支因子：分支因子是指可能接在任何上文之后的单词的数目。显然，如果我们的模型啥也没学习到，那么测试集任何单词可以接在任何上文之后，分支因子很高，困惑度很高；相反，如果我们的模型学习到了具体的规则，那么单词被限制接在一些指定上文之后，困惑度变低。困惑度使用了概率加权分支因子，分支因子的大小在模型学习前后不变，”morning”仍然可以接到任何上文之后，但是它接到”good”之后的概率变大了，因此是加权的分支因子。
  - 熵：对于语言序列，我们定义一个序列的熵为：    $$H(w_1,w_2,…,w_n )=-\sum _{W_1^n \in L} p(W_1^n) \log ⁡p(W_1^n)$$也就是这个序列中所有前缀子序列的熵之和，其均值是序列的熵率。计算整个语言的熵，假设语言是一个产生单词序列的随机过程，单词序列无限长，则其熵率是：$$H(L)=\lim _{n \rightarrow \infty}⁡ \frac 1n H(w_1,w_2,…,w_n) =\lim _{n \rightarrow \infty} -⁡\frac 1n \sum _{W \in L} p(W_1^n)  \log ⁡p(W_1^n)$$根据Shannon-McMillan-Breiman理论，在n趋于无穷的情况下，如果语言既是平稳又是正则的，上面这些子串的和的熵，可以用最大串代替每一个子串得到，这里的代替是指log后面求的是最大串的概率，log之前的概率依然是各个子串的概率？假如是这样的话提出最大串的概率对数，对所有子串概率求和得到：$$H(L)=\lim _{n \rightarrow \infty} -⁡ \frac 1n \log ⁡p(w_1,w_2,…,w_n)$$交叉熵可以衡量我们的模型生成的概率分布到指定概率分布之间的距离，我们希望模型生成概率分布尽可能近似真实分布，即交叉熵小。具体衡量时是对相同的语言序列，计算训练得到的模型m和理想模型p在生成这个序列上的概率的交叉熵：$$H(p,m) = \lim _{n \rightarrow \infty}⁡ - \frac 1n \sum _{W \in L} p(W_1^n) \log⁡ m(W_1^n)$$但是我们不知道理想的分布p，这时根据之前的Shannon-McMillan-Breiman定理，得到了只包含一个概率分布的序列交叉熵（？）：$$H(p,m)=\lim _{n \rightarrow \infty}⁡ - \frac 1n \log⁡ m(W_1^n)$$在测试数据上我们没有无限长的序列，就用有限长的序列的交叉熵近似这个无限长序列的交叉熵。困惑度则是这个（近似的？只包含一个概率分布的？）交叉熵取指数运算：
    
    $$
    Perplexity(W) = 2^{H(W)} \\
= P(w_1 w_2 ... w_N)^{\frac {-1}{N}} \\
= \sqrt[n]{\frac {1}{P(w_1 w_2 ... w_N)}} \\
= \sqrt[n]{\prod _{i=1}^N \frac {1}{P(w_i | w_1 ... w_{i-1})}} \\
    $$

## 平滑

- 因为N元语法模型依赖语料，一般而言对于N越高的N元语法，语料提供的数据越稀疏。这种情况下N元语法对于那些计数很小的语法估计很差，且如果测试集中某一句包含了训练集中没有出现的N元语法时，我们无法使用困惑度进行评价。因此我们使用平滑作为一种改进方法，使得N元语法的最大似然估计能够适应这些存在0概率的情况。
- 接下来介绍了两种平滑：
  - 拉普拉斯平滑（加1平滑）
  - Good-Turing 打折法

### 拉普拉斯平滑

- 加1平滑就是在计算概率归一化之前，给每个计数加1，对应的，归一化时分母整体加了一个词典大小:
  
  $$
  P_{Laplace}(w_i) = \frac {c_i + 1}{N+V} \\
  $$
- 为了表现平滑的作用，引入调整计数$c^{*}$，将平滑后的概率写成和平滑之前一样的形式：
  
  $$
  P_{Laplace} (w_i) = \frac {(C_i^{*})}{N} \\
C_i^{*} = \frac {(C_i+1)N}{(N+V)} \\
  $$
- 一种看待平滑的角度是：对每个非0计数打折，分一些概率给0计数，定义相对打折$d_c$（定义在非0计数上），
  
  $$
  d_c = \frac {c^{*}} {c}
  $$
- $d_c$代表了打折前后单词计数的变化。平滑之后，对于非0计数，当$C_i < \frac NV$时，计数增加；否则计数减少。计数越大，打折越多，增加越少（减少越多）。当0计数很多时，N/V较小，这时大部分非0计数都会减少，且减少较多。
- 而0计数则没有收到打折的影响。因此在一轮不同程度的增长之后，再归一化的结果就是非0计数分享了一些概率给0计数。写成调整计数的形式，就是非0计数减少数值，0计数变化（一般是减少）数值（但不是减少的完全等于增加的）。 书中给出了一个例子，下图是一部分语料的二元语法平滑之后的计数，蓝色代表平滑加1之后的0计数：
  ![FoZNDg.png](https://s2.ax1x.com/2019/01/03/FoZNDg.png)
  如果把表写成调整计数的形式：
  ![FoZtKS.png](https://s2.ax1x.com/2019/01/03/FoZtKS.png) 
- 可以看到，本来的0计数（蓝色）从0变大，而其他的计数减少，例如< i want>，从827减少到527，<want to>从608减少到238。
- 当0计数很多时，非0计数减少的数值很多，可以使用一个小于1的小数$\delta$代替1，即加$\delta$平滑。通常这个$\delta$是动态变化的。

### GT打折法

- 类似于Good-Turing打折法, Witten-Bell打折法， Kneyser-Ney 平滑一类的方法，它们的基本动机是用只出现一次的事物的计数来估计从未出现的事物的计数。只出现一次的语法称为单件（singleton）或者罕见语（hapax legomena）。Good-Turing打折法用单件的频率来估计0计数二元语法。
- 定义N_c为出现c次的N元语法的总个数（不是总个数乘以c），并称之为频度c的频度。对N_c中的c的最大似然估计是c。这样相当于将N元语法按其出现次数分成了多个桶，GT打折法用c+1号桶里语法概率的最大似然估计来重新估计c号桶内语法的概率。因此GT估计之后最大似然估计得到的c被替换成：
  
  $$
  c^{*}=(c+1) \frac {N_{c+1}}{N_c} 
  $$
- 之后计算某N元语法的概率：
  - 从未出现：$P_{GT}^{*}=\frac{N_1}{N}$。其中N是所有N元语法数$(\sum _i N_i * i)$。这里假设了我们已知$N_0$，则此式表示某一具体未知计数N元语法概率时还应除以$N_0$。
  - 已出现（已知计数）：$P_{GT}^{*} = \frac{c^{*}}{N}$
- 这样计算，$N_1$的一些概率转移到了$N_0$上。GT打折法假设所有的N元语法概率分布满足二项式分布，且假设我们已知$N_0$，以二元语法为例：
  
  $$
  N_0 = V^2 - \sum _{i>0} N_i \\ 
  $$
- 其他注意事项：
  - 有些$N_c$为0，这时我们无法用这些$N_c$来计算平滑后的c。这种情况下我们直接放弃平滑，令$c^{*} = c$，再根据正常的数据计算出一个对数线性映射，$log⁡(N_c) = a + b \log(c)$，代入放弃平滑的c并用其倒推计算计数为0的$N_c$，使得这些$N_c$有值，不会影响更高阶的c的计算。
  - 只对较小c的$N_c$进行平滑，较大c的$N_c$认为足够可靠，设定一个阈值k，对$c < k$的$N_c$计算：
    
    $$
    c^{*} = \frac {(c+1) \frac {N_c+1}{N_c} - c \frac {(k+1) N_{k+1} }{N_1} } {1- \frac {(k+1)N_{k+1}} {N_1}} \\
    $$
  - 计算较小的c如c=1时，也看成c=0的情况进行平滑
- 一个例子：
  ![FoZGgf.png](https://s2.ax1x.com/2019/01/03/FoZGgf.png)

## 插值与回退

- 上述的平滑只考虑了如何转移概率到计数为0的语法上去，对于条件概率$p(w|h)$，我们也可以采用类似的思想，假如不存在某个三元语法帮助计算$p(w_n |w_{n-1} w_{n-2})$，则可以用阶数较低的语法$p(w_n |w_{n-1})$帮助计算，有两种方案：
  - 回退：用低阶数语法的替代0计数的高阶语法
  - 插值：用低阶数语法的加权估计高阶语法
- 在Katz回退中，我们使用GT打折作为方法的一部分：GT打折告诉我们有多少概率可以从已知语法中分出来，Katz回退告诉我们如何将这些分出来的概率分配给未知语法。在之前的GT打折法中，我们将分出的概率均匀分给每一个未知语法，而Katz回退则依靠低阶语法的信息来分配：
  ![FoZJv8.png](https://s2.ax1x.com/2019/01/03/FoZJv8.png)
- 其中$P^{*}$是打折之后得到的概率；\alpha是归一化系数，保证分出去的概率等于未知语法分配得到的概率。
- 插值则是用低阶语法概率加权求和得到未知高阶语法概率：
  ![FoZUbQ.png](https://s2.ax1x.com/2019/01/03/FoZUbQ.png)
- 加权的系数还可以通过上下文动态计算。具体系数的计算有两种方法：
  - 尝试各种系数，用在验证集上表现最好的系数组合
  - 将系数看成是概率生成模型的隐变量，使用EM算法进行推断

## 实际问题：工具和数据格式

- 在语言模型计算中，一般将概率取对数进行计算，原因有二：防止数值下溢；取对数能将累乘运算变成累加，加速计算。
- 回退N元语法模型一般采用ARPA格式。ARPA格式文件由一些头部信息和各类N元语法的列表组成，列表中包含了该类N元语法下所有语法，概率，和回退的归一化系数。只有能够称为高阶语法前缀的低阶语法才能在回退中被利用，并拥有归一化系数。
- 两种计算N元语法模型的工具包：SRILM toolkit 和Cambridge-CMU toolkit

## 语言建模中的高级问题

### 高级平滑方法：Kneser-Ney平滑

- 注意到在GT打折法当中，打折之后估计的c值比最大似然估计得到的c值近似多出一个定值d。绝对打折法便考虑了这一点，在每个计数中减去这个d：
  ![FoZwUs.png](https://s2.ax1x.com/2019/01/03/FoZwUs.png)
- Kneser-Ney平滑吸收了这种观点，并且还考虑了连续性：在不同上文中出现的单词更有可能出现在新的上文之后，在回退时，我们应该优先考虑这种在多种上文环境里出现的词，而不是那些出现次数很多，但仅仅在特定上文中出现的词。
  ![FoZdEj.png](https://s2.ax1x.com/2019/01/03/FoZdEj.png)
- 在Kneser-Ney中，插值法能够比回退法取得更加好的效果：
  ![FoZ05n.png](https://s2.ax1x.com/2019/01/03/FoZ05n.png)

### 基于分类的N元语法

- 这种方法是为了解决训练数据的稀疏性。例如IBM聚类，每个单词只能属于一类，以二元语法为例，某个二元语法的条件概率的计算变为给定上文所在类，某个单词的条件概率，还可以进一步链式分解为两个类的条件概率乘以某个单词在给定其类条件下的条件概率：
  
  $$
  p(w_i│w_{i-1} ) \approx p(w_i│c_{i-1} ) = p(c_i |c_{i-1}) \cdot p(w_i |c_i)
  $$

### 语言模型适应和网络应用

- 适应是指在大型宽泛的语料库上训练语言模型，并在小的细分领域的语言模型上进一步改进。网络是大型语料库的一个重要来源。在实际应用时我们不可能搜索每一个语法并统计搜索得到所有页面上的所有语法，我们用搜索得到的页面数来近似计数。

### 利用更长距离的上文信息

- 通常我们使用二元和三元语法模型，但是更大的N能够带来更好的效果。为了捕捉更长距离的上文信息，有以下几种方法：
  - 基于缓存机制的N元语法模型
  - 基于主题建模的N元语法模型，对不同主题建模语言模型，再加权求和
  - 不一定使用相邻的上文信息，例如skip N-grams或者不一定使用定长的上文信息，例如变长N-grams

# 第十六章：语言的复杂性

## Chomsky层级

- Chomsky层级反映了不同形式化方法描述的语法之间的蕴含关系，较强生成能力或者说更复杂的语法在层级的外层。从外到内，加在可重写语法规则上的约束增加，语言的生成能力逐渐降低。
  ![FoZXad.png](https://s2.ax1x.com/2019/01/03/FoZXad.png)
- 五种语法对应的规则和应用实例：
  ![Foepxf.png](https://s2.ax1x.com/2019/01/03/Foepxf.png)
  - 0型语法：规则上只有一个限制，即规则左侧不能为空字符串。0型语法刻画了递归可枚举语言
  - 上下文相关语法：可以把上下文\alpha，\beta之间的非终极符号A重写成任意非空符号串
  - 温和的上下文相关语法
  - 上下文无关语法：可以把任何单独的非终极符号重写为由终极符号和非终极符号构成的字符串，也可以重写为空字符串
  - 正则语法：可以是右线性也可以是左线性，以右线性为例，非终极符号可以重写为左边加了若干终极符号的另一个非终极符号，右线性不断地在字符串左侧生成终极符号。

## 自然语言是否正则

- 判断语言是否正则能够让我们了解应该用哪一层次的语法来描述一门语言，且这个问题能够帮助我们了解自然语言的不同方面的某些形式特性。
- 抽吸引理：用来证明一门语言不是正则语言。
  - 如果一门语言可以被有限状态自动机来描述，则与自动机对应有一个记忆约束量。这个约束量对于不同的符号串不会增长的很大，因为其状态数目是固定的，更长的符号串应该是通过状态之间转移产生而不是增加状态数目。因此这个记忆量不一定和输入的长度成比例。
  - 如果一个正则语言能够描述任意长的符号序列，比自动机的状态数目还多，则该语言的自动机中必然存在回路。
    ![FoZxPI.png](https://s2.ax1x.com/2019/01/03/FoZxPI.png)
- 如图所示自动机，可以表述xyz,xyyz,xyyyz.....，当然也可以将中间无限长的y序列“抽吸掉”，表述xz。抽吸引理表述如下：
- 设L是一个有限的正则语言，那么必然存在符号串x,y,z,使得对于任意n≥0，y≠$\epsilon$，且xy^n z∈L
- 即假如一门语言是正则语言，则存在某一个符号串y，可以被适当的“抽吸”。这个定理是一门语言是正则语言的必要非充分条件。
- 有学者证明英语不是一门正则语言：
  - 具有镜像性质的句子通过抽吸原理可以证明不是正则语言，而英语中一个特殊的子集合和这种镜像性质的句子是同态的。
  - 另一种证明基于某些带有中心-嵌套结构的句子。这种句子可以由英语和某一类简单的正则表达式相交得到，通过抽吸原理可以得到这种句子不是正则语言。英语和正则语言的交不是正则语言，则英语不是正则语言。

## 自然语言是否上下文无关

- 既然自然语言不是正则语言，我们接着考虑更宽松的限定，自然语言是否是上下文无关的？
- 不是......

## 计算复杂性和人的语言处理

- 人对中心嵌套句子处理很困难，因为人们剖析时利用的栈记忆有限，且栈中不同层次记忆容易混淆。

# 第五章：词类标注

- 各种表述：POS（Part Of Speech）、word classes（词类）、morphological classes（形态类）、lexical tags（词汇标记）。
- POS的意义在于：
  - 能够提供关于单词及其上下文的大量信息。
  - 同一单词在不同词类下发音不同，因此POS还能为语音处理提供信息。
  - 进行词干分割（stemming），辅助信息检索
- 本章介绍三种词类标注算法：
  - 基于规则的算法
  - 基于概率的算法，隐马尔科夫模型
  - 基于变换的算法

## 一般词类

- POS分为封闭集和开放集，封闭集集合相对稳定，例如介词，开放集的词语则不断动态扩充，例如名词和动词。特定某个说话人或者某个语料的开放集可能不同，但是所有说一种语言以及各种大规模语料库可能共享相同的封闭集。封闭集的单词称为虚词（功能词，function word），这些词是语法词，一般很短，出现频次很高。
- 四大开放类：名词、动词、形容词、副词。
- 名词是从功能上定义的而不是从语义上定义的，因此名词一般表示人、地点、事物，但既不充分也不必要。定义名词：
  - 与限定词同时出现
  - 可以受主有代词修饰
  - 大多数可以以复数形式出现（即可数名词），物质名词不可数。单数可数名词出现时不能没有冠词
- 动词，表示行为和过程的词，包括第三人称单数、非第三人称单数、进行时、过去分词几种形态
- 形容词，描述性质和质量
- 副词，用于修饰，副词可以修饰动词、动词短语、其它副词。
- 英语中的一些封闭类：
  - 介词 prepositions：出现在名词短语之前，表示关系
  - 限定词 determiners 冠词 articles：与有定性（definiteness）相关
  - 代词 pronouns：简短的援引某些名词短语、实体、或事件的一种形式
  - 连接词 conjunctions：用于连接和补足（complementation）
  - 助动词 auxiliary verbs：标志主要动词的某些语义特征，包括：时态、完成体、极性对立、情态
  - 小品词 particles：与动词结合形成短语动词
  - 数词 numerals

## 词类标注

- 标注算法的输入是单词的符号串和标记集，输出要让每一个单词标注上一个单独且最佳的标记。如果每个单词只对应一种词性，那么根据已有的标记集，词类标注就是一个简单的查表打标的过程，但是很多词存在多种词性，例如book既可以是名词也可以是动词，因此要进行消歧，词类标注是歧义消解的一个重要方面。

## 基于规则的词类标注

- 介绍了ENGTWOL系统，根据双层形态学构建，对于每一个词的每一种词类分别立条，计算时不计屈折形式和派生形式.
- 标注算法的第一阶段是将单词通过双层转录机，得到该单词的所有可能词类
- 之后通过施加约束规则排除不正确的词类。这些规则通过上下文的类型来决定排除哪些词类。

## 基于隐马尔科夫模型的词类标注

- 使用隐马尔科夫模型做词类标注是一类贝叶斯推断，这种方法将词类标注看成是序列分类任务。观察量为一个词序列（比如句子），任务是给这个序列分配一个标注序列。
- 给定一个句子，贝叶斯推断想要在所有标注序列可能中选择最好的一个序列，即
  
  $$
  {t_1^n} _{best} = {argmax} _{t_1^n}  P(t_1^n |w_1^n)
  $$
- 使用贝叶斯法则将其转化为：
  
  $$
  {t_1^n} _{best}={argmax} _{t_1^n}  \frac{P(w_1^n│t_1^n)P(t_1^n)}{P(w_1^n)} = {argmax} _{t_1^n} P(w_1^n│t_1^n)P(t_1^n)
  $$
- 隐马尔科夫模型在此基础上做了两点假设
  - 一个词出现的概率只与该词的词类标注有关，与上下文其他词和其他标注无关，从而将序列的联合概率拆解为元素概率之积，即：P(w_1^n│t_1^n) \approx \prod _{i=1}^n P(w_i |t_i)
  - 一个标注出现的概率只与前一个标注相关，类似于二元语法的假设：P(t_1^n ) \approx \prod _{i=1}^n P(t_i |t_{i-1})
- 在两种假设下简化后的最好标注序列表达式为：
  
  $$
  {t_1^n}_{best} = {argmax} _{t_1^n} P(t_1^n│w_1^n) \approx {argmax} _{t_1^n} \prod _{i=1}^n P(w_i│t_i) P(t_i |t_{i-1})
  $$
- 上面这个概率表达式实际上将HMM模型的联合概率拆成了各个部分转移概率的乘积，具体而言分为标签转移概率（隐变量之间转移）和词似然（隐变量转移到可观察变量）。通过最大似然估计，我们可以通过古典概型的方法从已标注的语料中计算出这两类概率：
  
  $$
  P(t_i│t _{i-1} ) = (C(t _{i-1},t_i))/C(t _{i-1} ) \\
P(w_i│t_i ) = \frac{C(t_i,w_i)}{C(t_i)} \\
  $$
- 一个例子：HMM模型如何正确的将下句中的race识别为动词而不是名词：
- Secretariat is expected to race tomorrow.
- 画出上句中race被识别为动词和名词两种情况下的HMM模型，可以看到两个模型对比只有三个转移概率不同，用加粗线标出：
  ![FoZDCq.png](https://s2.ax1x.com/2019/01/03/FoZDCq.png)
- HMM词类标注器消歧的方式是全局的而不是局部的。我们在语料中统计得到这三种转移概率，再累乘，结果是(a)的概率是(b)概率的843倍。显然race应该被标注为动词。

## 形式化隐马尔科夫模型标注器

- HMM模型是有限自动机的扩展，具体而言是一种加权有限自动机，马尔可夫链的扩展，这种模型允许我们考虑观察量和隐变量，考虑包含隐变量的概率模型。HMM包含以下组件：
  - Q：大小为N的状态集
  - A：大小为N*N的转移概率矩阵
  - O：大小为T的观察事件集
  - B：观察似然序列，又叫发射概率，$b_i (o_t)$描述了从状态i里生成观察o_t的概率
  - $q_0，q_F$：特殊的起始状态和最终状态，没有相连接的观察量
- A中的概率和B中的概率对应着之前式子中每一个累乘项里的先验$P(w_i│t_i )$和似然$P(t_i |t _{i-1})$概率：
  
  $$
  {t_1^n}_{best}={argmax} _{t_1^n} P(t_1^n│w_1^n ) \approx {argmax} _{t_1^n} \prod _{i=1}^n P(w_i│t_i)P(t_i |t _{i-1})
  $$

## HMM标注的维特比算法

- 在HMM模型中，已知转移概率和观察序列，求隐变量的任务叫做解码。解码的一种算法即维特比算法，实质上是一种动态规划算法，与之前求最小编辑距离的算法类似。
- 首先我们从语料中计算得到A和B两个矩阵，即模型的转移概率已知，对于给定的观察序列，按照以下步骤执行维特比算法：
  ![FoZyvT.png](https://s2.ax1x.com/2019/01/03/FoZyvT.png)
- 算法维护一个$(N+2)*T$的概率矩阵viterbi，加了2代表初始状态和结束状态，viterbi[s,t]代表了在第t步状态为s时的最佳路径概率，而backpointer[s,t]对应着保存了该最佳路径的上一步是什么状态，用于回溯输出整个最佳路径。
- 关键的转移在于$viterbi[s,t] \leftarrow max _{s^{*}=1}^N⁡ viterbi[s^{*},t-1] * a_{s^{*},s} * b_s (o_t)$即当前时间步最佳路径是由上一时间步各个状态的最佳路径转移过来的，选择上一步最佳路径概率与转移概率乘积最大的路径作为当前时间步的最佳路径。从动态规划的角度而言，即长度为t的最佳路径，必定是从长度为t-1的最佳路径里选择一条转移得到，否则肯定可以从另一条概率更大的路径转移获得更优解。这样就限制了最佳路径的生成可能，减少了计算量。

## 将HMM算法扩展到三元语法

- 现代的HMM标注器一般在标注转移概率上考虑更长的上文历史：
  
  $$
  P(t_1^n ) \approx \prod_{i=1}^n P(t_i |t _{i-1},t_{i-2})
  $$
- 这样的话需要在序列开头和结尾做一些边界处理。使用三元语法的一个问题是数据稀疏：例如我们从没有在训练集中见过标注序列PRP VB TO，则我们无法计算P(TO|PRP,VB)。一种解决办法是线性插值：
  
  $$
  P(t_i│t _{i-1} t _{i-2} ) = \lambda _1 P ̂(t_i│t _{i-1} t _{i-2} )+\lambda _2 P ̂(t_i│t _{i-1} )+\lambda _3 P ̂(t_i)
  $$
- 使用删除插值的办法确定系数$\lambda$：
  ![FoZr80.png](https://s2.ax1x.com/2019/01/03/FoZr80.png)

## 基于变换的标注

- 基于变换的方法结合了基于规则和基于概率方法的优点。基于变换的方法依然需要规则，但是从数据中总结出规则，是一种监督学习方法，称为基于变换的学习（Transformation Based Learning，TBL）。在TBL算法中，语料库首先用比较宽的规则来标注，然后再选择稍微特殊的规则来修改，接着再使用更窄的规则来修改数量更少的标记。

## 如何应用TBL规则

- 首先应用最宽泛的规则，就是根据概率给每个词标注，选择概率最大的词类作为标注。之后应用变换规则，即如果满足某一条件，就将之前标注的某一词类变换（纠正）为正确的词类，之后不断应用更严格的变换，在上一次变换的基础上进行小部分的修改。
- 如何学习到TBL规则
  - 首先给每个词打上最可能的标签
  - 检查每一个可能的变换，选择效果提升最多的变换，此处需要直到每一个词正确的标签来衡量变换带来的提升效果，因此是监督学习。
  - 根据这个被选择的变换给数据重新打标，重复步骤2，直到收敛（提升效果小于某一阈值）
- 以上过程输出的结果是一有序变换序列，用来组成一个标注过程，在新语料上应用。虽然可以穷举所有的规则，但是那样复杂度太高，因此我们需要限制变换集合的大小。解决方案是设计一个小的模板集合（抽象变换）,每一个允许的变换都是其中一个模板的实例化。

## 评价和错误分析

- 一般分为训练集、验证集、测试集，在训练集内做十折交叉验证。
- 与人类标注的黄金标准比较计算准确率作为衡量指标。
- 一般用人类表现作为ceiling，用一元语法最大概率标注的结果作为baseline。
- 通过含混矩阵或者列联表来进行错误分析。在N分类任务中，一个N*N的含混矩阵的第i行第j列元素指示第i类被错分为第j类的次数在总分错次数中的占比。一些常见的容易分错的词性包括：
  - 单数名词、专有名词、形容词
  - 副词、小品词、介词
  - 动词过去式、动词过去分词、形容词

## 词性标注中的一些其他问题

- 标注不确定性：一个词在多个词性之间存在歧义，很难区分。这种情况下有些标注器允许一个词被打上多个词性标注。在训练和测试的时候，有三种方式解决这种多标注词：
  - 通过某种方式从这些候选标注中选择一个标注
  - 训练时指定一个词性，测试时只要打上了候选词性中任意一个就认为标注正确
  - 将整个不确定的词性集看成一个新的复杂词性
- 多部分词：在标注之前需要先分词，一些多部分词是否应该被分为一部分，例如New York City应该分成三部分还是一个整体，也是各个标注系统需要考虑的。
- 未知词：不在词典中的词称为未知词。对于未知词，训练集无法给出它的似然P(w_i |t_i)，可以通过以下几种方式解决：
  - 只依赖上下文的POS信息预测
  - 用只出现一次的词来估计未知词的分布，类似于Good Turing打折法
  - 使用未知词的单词拼写信息，正词法信息。例如连字符、ed结尾、首字母大写等特征。之后在训练集中计算每个特征的似然，并假设特征之间独立，然后累乘特征似然作为未知词的似然：$P(w_i│t_i )=p(unknown word│t_i ) * p(capital│t_i ) * p(endings/hyph|t_i)$
  - 使用最大熵马尔可夫模型
  - 使用对数线性模型

## 噪声信道模型

- 贝叶斯推断用于标注可以认为是一种噪声信道模型的应用，本节介绍如何用噪声信道模型来完成拼写纠正任务。
  之前对于非单词错误，通过词典查找可以检测到错误，并根据最小编辑距离纠正错误，但这种方法对于真实单词错误无能为力。噪声信道模型可以纠正这两种类型的拼写错误。
- 噪声信道模型的动机在于将错误拼写的单词看成是一个正确拼写的单词经过一个噪声信道时受到干扰扭曲得到。我们尝试所有可能的正确的词，将其输入信道，最后得到的干扰之后的词与错误拼写的词比较，最相似的例子对应的输入词就认为是正确的词。这类噪声信道模型，比如之前的HMM标注模型，是贝叶斯推断的一种特例。我们看到一个观察两（错误拼写词）并希望找到生成这个观察量的隐变量（正确拼写词），也就是找最大后验。
- 将噪声信道模型应用于拼写纠正：首先假设各种拼写错误类型，错拼一个、错拼两个、漏拼一个等，然后产生所有可能的纠正，除去词典中不存在的，最后分别计算后验概率，选择后验概率最大的作为纠正。其中需要根据局部上下文特征来计算似然。
- 另一种纠正算法是通过迭代来改进的方法：先假设拼写纠正的含混矩阵是均匀分布的，之后根据含混矩阵运行纠正算法，根据纠正之后的数据集更新含混矩阵，反复迭代。这种迭代的算法是一种EM算法。

## 根据上下文进行拼写纠正

- 即真实单词拼写错误的纠正。为了解决这类任务需要对噪声信道模型进行扩展：在产生候选纠正词时，需要包括该单词本身以及同音异形词。之后根据整个句子的最大似然来选择正确的纠正词。

# 第六章：隐马尔科夫模型和最大熵模型

- 隐马尔科夫模型用来解决序列标注（序列分类问题）。
- 最大熵方法是一种分类思想，在满足给定条件下分类应满足限制最小（熵最大），满足奥卡姆剃刀原理。
- 最大熵马尔可夫模型是最大熵方法在序列标注任务上的扩展。

## 马尔可夫链

- 加权有限自动状态机是对有限自动状态机的扩展，每条转移路径上加上了概率作为权重，说明从这条路径转移的可能性。马尔可夫链是加权有限状态自动机的一种特殊情况，其输入序列唯一确定了自动机会经过的状态序列。马尔可夫链只能对确定性序列分配概率。
- 我们将马尔可夫链看作一种概率图模型，一个马尔可夫链由下面的成分确定：
  
  $$
  Q=q_1 q_2…q_N \\
A=a_{01} a_{02} … a_{n1} … a_{nn} \\
q_0,q_F \\
  $$
- 分别是
  - 状态集合
  - 转移概率矩阵，其中a_ij代表了从状态i转移到状态j的概率$P(q_j |q_i)$
  - 特殊的开始状态和结束状态
- 概率图表示将状态看成图中的点，将转移看成边。
- 一阶马尔可夫对转移做了很强的假设：某一状态的概率只与前一状态相关：
  
  $$
  P(q_i│q_1…q _{i-1} )=P(q_i |q _{i-1})
  $$
- 马尔可夫链的另一种表示不需要开始和结束状态：
  
  $$
  \pi = \pi _1,\pi _2 , … , \pi _N \\
QA={q_x,q_y…} \\
  $$
- 分别是：
  - 状态的初始概率分布，马尔可夫链以概率$\pi _i$从状态i开始
  - 集合QA是Q的子集，代表合法的接受状态
- 因此状态1作为初始状态的概率既可以写成$a_{01}$也可以写成$\pi _1$。

## 隐马尔科夫模型

- 当马尔可夫链已知时，我们可以用其计算一个观测序列出现的概率。但是观测序列可能依赖于一些不可观测的隐变量，我们可能感兴趣的是推断出这些隐变量。隐马尔科夫模型允许我们同时考虑观测变量和隐变量。
- 如之前一样定义隐马尔科夫模型：
  - Q：大小为N的状态集
  - A：大小为N*N的转移概率矩阵
  - O：大小为T的观察事件集
  - B：观察似然序列，又叫发射概率，$b_i (o_t)$描述了从状态i里生成观察$o_t$的概率
  - $q_0，q_F$：特殊的起始状态和最终状态，没有相连接的观察量
- 同样的，隐马尔科夫也可以用另一种不依赖初始和结束状态的方式表示。隐马尔科夫模型也做了两个假设，分别是隐状态之间转移和隐状态到观察量转移的一阶马尔可夫性。
- 对于隐马尔科夫模型需要解决三类问题：
  - 似然计算：已知参数和观测序列，求似然$P(O|\lambda)$
  - 解码：已知参数和观测序列，求隐状态序列
  - 学习：已知观测序列和隐状态集合，求解模型参数

## 计算似然：前向算法

- 对于马尔可夫链，其没有隐状态到观测量的转移概率矩阵，可以看成观察量与隐状态相同。在隐马尔科夫模型中不能直接计算似然，我们需要直到隐状态序列。
- 先假设隐状态序列已知，则似然计算为：
  
  $$
  P(O│Q) = \prod _{i=1}^T P(o_i |q_i)
  $$
- 根据隐状态转移的一阶马尔可夫性，可以求得隐状态的先验，乘以似然得到观测序列和隐状态序列的联合概率：
  
  $$
  P(O,Q)=P(O│Q) * P(Q) = \prod _{i=1}^n P(o_i│q_i )  \prod _{i=1}^n P(q_i |q _{i-1})
  $$
- 对于联合概率积分掉隐状态序列，就可以得到观测概率的似然：
  
  $$
  P(O) = \sum _Q P(O,Q) = \sum _Q P(O|Q)P(Q) 
  $$
- 这样计算相当于考虑了所有的隐状态可能，并对每一种可能从隐状态序列开始到结束计算一次似然，实际上可以保留每次计算的中间状态来减少重复计算，也就是动态规划。在前向计算HMM观测似然使用的动态规划算法称为前向算法：
  - 令$\alpha _t (j)$代表在得到前t个观测量之后当前时刻隐变量处于状态j的概率,\lambda为模型参数：
    
    $$
    \alpha _t (j) = P(o_1,o_2…o_t,q_t=j|\lambda)
    $$
  - 这个概率值可以根据前一时间步的\alpha值计算出来，避免了每次从头开始计算：
    
    $$
    \alpha _t (j) = \sum _{i=1}^N \alpha _{t-1} (i) a_{ij} b_j (o_t)
    $$
  - 初始化$\alpha _1 (j)$：
    
    $$
    \alpha _1 (j)=a_{0s} b_s (o_1)
    $$
  - 终止状态：
    
    $$
    P(O│\lambda) = \alpha _T (q_F) = \sum _{i=1}^N \alpha _T (i) \alpha _{iF}
    $$

## 解码：维特比算法

- 解码任务是根据观测序列和参数推断出最有可能隐状态序列。最朴素的做法：对于每种可能的隐状态序列，计算观测序列的似然，取似然最大时对应的隐状态序列。但是这样做就如同朴素的计算似然方法一样，时间复杂度过高，同样的，我们使用动态规划来缩小求解的规模。在解码时使用了一种维特比算法。
  - 令$v_t (j)$代表已知前t个观测量（1~t）和已知前t个隐状态（0~t-1）的条件下，当前时刻隐状态为j的概率：
    
    $$
    v_t (j)=max _{q_0,q_1,…,q_{t-1}} P(q_0,q_1…q_{t-1},o_1,o_2 … o_t,q_t=j|\lambda)
    $$
  - 其中我们已知了前t个时间步最大可能的隐状态序列，这些状态序列也是通过动态规划得到的：
    
    $$
    v_t (j)=max _{i=1}^N⁡ v_{t-1} (i) a_{ij} b_j (o_t)
    $$
  - 为了得到最佳的隐状态序列，还需要记录每一步的最佳选择，方便回溯得到路径：
    
    $$
    {bt}_t (j) = argmax _{i=1}^N v_{t-1} (i) a_{ij} b_j (o_t)
    $$
  - 初始化：
    
    $$
    v_1 (j) = a_{0j} b_j (o_1) \ \  1 \leq j \leq N \\
{bt}_1 (j) = 0 \\
    $$
  - 终止，分别得到最佳隐状态序列（回溯开始值）及其似然值：
    
    $$
    P * = v_t (q_F ) = max_{i=1}^N⁡ v_T (i) * a_{i,F} \\
q_{T*} = {bt}_T (q_F ) = argmax _{i=1}^N v_T (i) * a_{i,F} \\
    $$
- 维特比算法减小时间复杂度的原因在于其并没有计算所有的隐状态路径，而是利用了每一时间步的最佳路径只能从上一时间步的最佳路径中延伸而来这一条件，减少了路径候选，避免了许多不必要的路径计算。并且每一步利用上一步的结果也是用了动态规划的思想减少了计算量。

## 训练隐马尔科夫模型：前向后向算法

- 学习问题是指已知观测序列和隐状态集合，求解模型参数。
- 前向后向算法，又称Baum-Welch算法，是EM算法的一种特例，用来求解包含隐变量的概率生成模型的参数。该算法通过迭代的方式反复更新转移概率和生成概率，直到收敛。BW算法通过设计计数值之比作为隐变量，将转移概率矩阵和生成概率矩阵一起迭代更新。
- 先考虑马尔科夫链的学习问题。马尔科夫链可以看作是退化的隐马尔科夫模型，即每个隐变量只生成和自己一样的观测量，生成其他观测量的概率为0。因此只需学习转移概率。
- 对于马尔可夫链，可以通过古典概型统计出转移概率：
  
  $$
  a_{ij} = \frac {Count(i \rightarrow j)} {\sum _{q \in Q} Count(i \rightarrow q)}
  $$
- 我们可以这样直接计算概率是因为在马尔可夫链中我们知道当前所处的状态。对于隐马尔科夫模型我们无法这样直接计算是因为对于给定输入，隐状态序列无法确定。Badum-Welch算法使用了两种简洁的直觉来解决这一问题：
  - 迭代估计，先假设一种转移概率和生成概率，再根据假设的概率推出更好的概率
  - 计算某一观测量的前向概率，并将这个概率分摊到不同的路径上，通过这种方式估计概率
- 首先类似于前向概率，我们定义后向概率：
  - 令$\beta _t (i)$代表在得到后t个观测量之后当前时刻隐变量处于状态i的概率,$\lambda$为模型参数：
    
    $$
    \beta _t (i) = P(o_{t+1},o_{t+2}…o_T,q_t=i|\lambda)
    $$
  - 类似于后向概率的归纳计算：
    
    $$
    \beta_t (i) = \sum _{j=1}^N a_{ij} b_j (o_{t+1} ) \beta _{t+1} (j),  \ \   1≤i≤N,1≤t<T
    $$
  - 初始化$\alpha _1 (j)$：
    
    $$
    \beta _T (i)=\alpha _(i,F)
    $$
  - 终止状态：
    
    $$
    P(O│\lambda)=\alpha _t (q_F )=\beta_1 (0)= \sum _{i=1}^N a_{0j} b_j (o_1) \beta _1 (j)
    $$
- 类似的，我们希望马尔可夫链中的古典概率能帮助我们估计转移概率：
  
  $$
  a_{ij}^{*} = \frac{从状态i转移到状态j的计数值期望}{从状态i转移出去的计数值期望}
  $$
- 如何估计计数值：我们将整个序列的转移路径计数值转化为时间步之间转移路径计数值之和，时间步之间某一条转移路径的概率为：
  
  $$
  P(q_t=i,q_{t+1}=j)
  $$
- 首先考虑所有的观测序列和这一转移路径的联合概率（省略了以参数$\lambda$为条件）：
  
  $$
  P(q_t=i,q_{t+1}=j,O)
  $$
- 观察下面的概率图：
  ![FoZWVJ.png](https://s2.ax1x.com/2019/01/03/FoZWVJ.png)
- 可以看到这一联合概率包含了三个部分：
  - T时刻隐状态为i的前向概率
  - T+1时刻隐状态为j的后向概率
  - T时刻与T+1时刻的状态转移概率以及生成对应观测量的生成概率
- 所以有：
  
  $$
  P(q_t=i,q_{t+1}=j,O)=\alpha _t (i) a_{ij} b_j (o_{t+1} ) \beta _{t+1} (j)
  $$
- 为了从联合分布中得到已知观测序列求转移路径的联合概率，需要计算观测序列的概率，可以通过前向概率或者后向概率求得：
  
  $$
  P(O)=\alpha _t (N)=\beta _T (1) = \sum _{j=1}^N \alpha _t (j) \beta_t (j)
  $$
- 最终得到
  
  $$
  ξ_t (i,j)=P(q_t=i,q_{t+1}=j│O) = \frac {(\alpha _t (i) a_{ij} b_j (o_{t+1} ) \beta_{t+1} (j))}{(\alpha _t (N))}
  $$
- 最后，对所有时间步求和就可以得到从状态i转移到状态j的期望计数值，从而进一步得到转移概率的估计：
  
  $$
  a_{ij}^{*} = \frac {\sum _{t=1}^{T-1} ξ_t (i,j)}{\sum _{t=1}^{T-1} \sum _{j=1}^{N-1} ξ_t (i,j)}
  $$
- 同样的，我们还希望得到生成概率的估计：
  
  $$
  b_{j}^{*} (v_k) = \frac {在状态j观测到符号v_k 的计数值期望}{状态j观测到所有符号的计数值期望}
  $$
- 类似的，通过先计算联合分布再计算条件分布的方式得到在t时刻处于隐状态j的概率：
  
  $$
  γ_t (j)=P(q_t=j│O) = \frac {P(q_t=j,O)}{P(O)}
  $$
- 联合概率包含两个部分，即t时刻处于状态j的前向概率和后向概率，所以有：
  
  $$
  γ_t (j) = \frac {\alpha _t (j) \beta_t (j)}{\alpha _t (N)}
  $$
- 类似的，对所有时间步累加，进而得到生成概率的估计：
  
  $$
  b_{j}^{*} (v_k) = \frac{\sum _{t=1 s.t. O_t=v_k}^T   γ_t (j) }{\sum _{t=1}^T   γ_t (j) }
  $$
- 这两个式子是在已知前向概率和后向概率$(\alpha,\beta)$的情况下，计算出中间变量（隐变量）(ξ,γ),引入隐变量的动机是将a、b估计值的期望计数值之比转化为概率之比，且这两个隐变量可以用a,b表示。再由隐变量计算出转移概率和生成概率，因此形成了一个迭代的循环，可以用EM算法求解：
  
  $$
  a,b→\alpha,\beta→ξ,γ→a,b
  $$
- E-step:
  
  $$
  γ_t (j) = (\alpha _t (j) \beta_t (j))/(\alpha _t (N)) ξ_t (i,j) \\
= (\alpha _t (i) a_{ij} b_j (o_{t+1} ) \beta_{t+1} (j))/(\alpha _t (N)) \\
  $$
- M-step（最大化的目标是什么）:
  
  $$
  a _{ij} = (\sum _{t=1}^{T-1}   ξ_t (i,j)  )/(\sum _{t=1}^{T-1} \sum _{j=1}^{N-1}   ξ_t (i,j)  ) \\
b ̂_j(v_k) = (\sum _{t=1 s.t. O_t=v_k}^T   γ_t (j) )/(\sum _{t=1}^T   γ_t (j) ) \\
  $$
- 迭代时需重新计算：
  
  $$
  \alpha _t (j) = \sum _{i=1}^N   \alpha_{t-1} (i) a_ij b_j (o_t) \\
\beta_t (i) = \sum _{j=1}^N   a_ij b_j (o_{t+1} ) \beta_{t+1} (j)  \\
  $$
- 迭代的初始状态对于EM算法来说很重要，经常是通过引入一些外部信息来设计一个好的初始状态。

## 最大熵模型：背景

- 最大熵模型另一种广为人知的形式是多项Logistic回归（Softmax?）。
- 最大熵模型解决分类问题，最大熵模型作为一种概率分类器，能够根据样本的特征求出样本属于每一个类别的概率，进而进行分类。
- 最大熵模型属于指数家族（对数线性）分类器，通过将特征线性组合，取指数得到分类概率：
  
  $$
  p(c│x)=\frac 1Z exp⁡(\sum _i   weight_i feature_i) 
  $$
- Z是一个归一化系数，使得生成的概率之和为1。

## 最大熵建模

- 将二分类Logistic回归推广到多分类问题就得到：
  
  $$
  P(c│x) = \frac {exp⁡(\sum _(i=0)^N   w_ci f_i) } {\sum _{c^{*} in C}   exp⁡(\sum _{i=0}^N   w_{c^{*} i} f_i)  }
  $$
- 语音和语言处理中的特征通常是二值的（是否有该特征），因此使用指示函数表示特征
  
  $$
  P(c│x) = \frac {exp⁡(\sum _{i=0}^N   w_{c_i} f_i (c,x)) }{\sum _{c^{*} \in C}   exp⁡(\sum _{i=0}^N   w_{c^{*} i} f_i (c^{*},x))  }
  $$
- 注意到在该模型中每一个类都有其独立的线性权重w_c。相比于硬分布，最大熵模型能够给出分到每一类的概率，因此可以求出每一时刻的分类概率进而求出整体分类概率，得到全局最优分类结果。注意到不同于支持向量机等模型，最大熵模型无法利用特征之间的组合，必须手动构造组合作为新的特征。
- 一般使用加了正则化的最大似然作为优化的目标函数：
  
  $$
  w ̂={argmax} _w \sum _i   \log P(y^{(i)}│x^{(i) } ) - \alpha \sum _{j=1}^N w_j^2  
  $$
- 这种正则化相当于给权重的概率分布加了一个零均值高斯先验，权重越偏离均值，即权重越大，其概率越低。
- 为什么多分类Logistic回归是最大熵模型：最大熵模型保证在满足给定约束下，无约束的部分分类应该是等概率分配，例如在两个约束下：
  
  $$
  P(NN)+P(JJ)+P(NNS)+P(VB)=1 \\
P(t_i=NN or t_i=NNS)=8/10 \\
  $$
- 则满足这两个约束，最大熵模型分配的概率结果为：
  
  $$
  p(NN)=4/10  \\
p(JJ)=1/10  \\
p(NNS)=4/10  \\
p(VB)=1/10 \\
  $$
- 在The equivalence of logistic regression and maximum entropy models一文中证明了在广义线性回归模型的平衡条件约束下，满足最大熵分布的非线性激活函数就是sigmoid，即logistic回归。

## 最大熵马尔可夫模型

- 最大熵模型只能对单一观测量分类，使用最大熵马尔可夫模型可以将其扩展到序列分类问题上。
- 最大熵马尔可夫比隐马尔科夫模型好在哪儿？隐马尔科夫模型对于每个观测量的分类依赖于转移概率和生成概率，假如我们想要在标注过程中引入外部知识，则需要将外部知识编码进这两类概率中，不方便。最大熵马尔可夫模型能够更简单的引入外部知识。
- 在隐马尔科夫模型中我们优化似然，并且乘以先验来估计后验：
  
  $$
  T ̂= {argmax}_T ∏_i   P(word_i│tag_i ) ∏_i   P(tag_i│tag _{i-1} )   
  $$
- 在最大熵隐马尔科夫模型中，我们直接计算后验。因为我们直接训练模型来分类，即最大熵马尔可夫模型是一类判别模型，而不是生成模型：
  
  $$
  T ̂= {argmax}_T ∏_i   P(tag_i |word_i,tag _{i-1}) 
  $$
- 因此在最大熵隐马尔科夫模型中没有分别对似然和先验建模，而是通过一个单一的概率模型来估计后验。两者的区别如下图所示：
  ![FoZgrF.png](https://s2.ax1x.com/2019/01/03/FoZgrF.png) 
- 另外最大熵马尔可夫模型可以依赖的特征更多，依赖方式更灵活，如下图：
  ![FoZcKU.png](https://s2.ax1x.com/2019/01/03/FoZcKU.png)
- 用公式表示这一差别：
  
  $$
  HMM:P(Q│O)=∏_{i=1}^n   P(o_i |q_i)×∏_{i=1}^n   P(q_i |q _{i-1})  \\
MEMM:P(Q│O)=∏_{i=1}^n   P(q_i |q _{i-1},o_i) \\
  $$
- 当估计单一转移概率（从状态q^{*}转移到状态q，产生观测量o）时，我们使用以下的最大熵模型：
  
  $$
  P(q│q^{*},o)=\frac{1}{Z(o,q^{*})} exp⁡(\sum _i   w_i f_i (o,q)) 
  $$

## 最大熵马尔可夫模型的解码（推断）

- MEMM同样使用维特比算法进行解码
- 使用维特比算法解码的通用框架是：
  
  $$
  v_t (j)=max_{i=1}^N⁡  v_{t-1} (i)P(s_j│s_i )P(o_t |s_j) 
  $$
- 在HMM模型中这一框架具体化为：
  
  $$
  v_t (j)=max_{i=1}^N⁡  v_{t-1} (i) a_ij b_j (o_t) 
  $$
- 在MEMM中直接将似然和先验替换为后验：
  
  $$
  v_t (j)=max_{i=1}^N⁡  v_{t-1} (j)P(s_j |s_i,o_t) 
  $$

## 最大熵马尔可夫模型的训练

- MEMM作为最大熵模型的推广，训练过程使用和最大熵模型一样的监督算法。如果训练数据的标签序列存在缺失，也可以通过EM算法进行半监督学习。

# 第十二章：英语的形式语法

## 组成性

- 英语中的单词是如何组成一个词组的呢？
- 换句话说，我们如何判断一些单词组合成了一个部分？一种可能是这种组合都能在相似的句法环境中出现，例如名词词组都能在一个动词之前出现。另一种可能依据来自于前置和后置结构，例如前置短语on September seventeenth可以放在句子的前面，中间或者后面，但是组合成这个短语的各个部分不能拆出来放在句子的不同位置，因此我们判断on September seventeenth这三个词组成了一个短语。

## 上下文无关法则

- 上下文无关语法，简称CFG，又称为短语结构语法，其形式化方法等价于Backus-Naur范式。一个上下文无关语法包含两个部分：规则或者产生式，词表。
- 例如，用上下文无关语法描述名词词组，一种描述方式是名词词组可以由一个专有名词构成，也可以由一个限定词加一个名词性成分构成，而名词性成分可以是一个或多个名词，此CFG的规则为：
  - NP→Det Nominal
  - NP→ProperNoun
  - Nominal→Noun|Noun Nominal
- CFG可以层级嵌套，因此上面的规则可以与下面表示词汇事实的规则（词表）结合起来：
  - Det→a
  - Det→the
  - Noun→flight
- 符号分为两类：
  - 终极符号：与现实中单词对应的符号，词表是引入终极符号的规则的集合
  - 非终极符号：表示终极符号的聚类或者概括性符号
- 在每个规则里箭头右边包含一个或多个终极符号和非终极符号，箭头左边为一个非终极符号，与每个单词相关联的是其词类范畴（词类）。
- CFG既可以看成是生成句子的一种机制，也可以看成是给一个句子分配结构的机制。
- 以之前提到的CFG为例，对一个符号串NP，可以逐步生成：
  
  $$
  NP→Det Nominal→Det Noun→a flight
  $$
- 称 a flight是NP的一个推导，一般用一个剖析树表示一种推导：
  ![FoZ5P1.png](https://s2.ax1x.com/2019/01/03/FoZ5P1.png)
  一个CFG定义了一个形式语言，形式语言是符号串的集合，如果有一个语法推导出的句子处于由该语法定义的形式语言中，这个句子就是合语法的。使用形式语言来模拟自然语言的语法成为生成式语法。
- 上下文无关语法的正式定义：
  - N：非终止符号（或者变量）的集合
  - Sigma：终止符号的集合，与N不相交
  - R：规则或者产生式的集合
  - S：指定的开始符号
- 一些约定定义：
  - 大写字母：代表非终止符号
  - S：开始符号
  - 小写希腊字母：从非终止符号和终止符号的并集中抽取出来的符号串
  - 小写罗马字母：终止符号串
- 直接导出的定义：
  **公式待补充**
- 导出是直接导出的泛化。之后我们可以正式定义由语法G生成的语言L是一个由终止符号组成的字符串集合，这些终止符号可以从指定的开始符号S通过语法G导出：
  **公式待补充**
- 将一个单词序列映射到其对应的剖析树成为句法剖析。

## 英语的一些语法规则

- 英语中最常用最重要的四种句子结构：
  - 陈述式结构：主语名词短语加一个动词短语
  - 命令式结构：通常以一个动词短语开头，并且没有主语
  - Yes-no疑问式结构：通常用于提问，并且以一个助动词开头，后面紧跟一个主语NP，再跟一个VP
  - Wh疑问式结构：包含一个wh短语成分
- 在之前的描述中开始符号用于单独生成整个句子，但是S也可以出现在语法生成规则的右边，嵌入到更大的句子当中。这样的S称为从句，拥有完整的语义。拥有完整的语义是指这个S在整体句子的语法剖析树当中，其子树当中的主要动词拥有所需的所有论元。

## 名词短语

- 限定词Det：名词短语可以以一些简单的词法限定词开始，例如a,the,this,those,any,some等等，限定词的位置也可以被更复杂的表示替代，例如所有格。这样的表示是可以递归定义的，例如所有格加名词短语可以构成更大的名词短语的限定词。在复数名词、物质名词之前不需要加限定词。
- 名词性词Nominal：包含一些名词前或者名词后修饰语
- 名词之前，限定词之后：一些特殊的词类可以出现在名词之前限定词之后，包括基数词Card、序数词Ord、数量修饰语Quant。
- 形容词短语AP：形容词短语之前可以出现副词
- 可以讲名词短语的前修饰语规则化如下（括号内代表可选）：
- NP->(Det)(Card)(Ord)(Quant)(AP)Nominal
- 后修饰语主要包含三种：
  - 介词短语PP：Nominal->Nominal PP(PP)(PP)
  - 非限定从句：动名词后修饰语GerundVP,GerundVP->GerundV NP | GerundV PP | GerundV | GerundV NP PP
  - 关系从句：以关系代词开头的从句 Nominal ->Nominal RelCaluse;RelCaluse -> (who|that) VP

## 一致关系

- 每当动词有一个名词作为它的主语时，就会发生一致关系的现象，凡是主语和他的动词不一致的句子都是不合语法的句子，例如第三人称单数动词没有加-s。可以使用多个规则的集合来扩充原有的语法，使得语法可以处理一致关系。例如yes-no疑问句的规则是
  
  $$
  S \rightarrow Aux \ NP \ VP
  $$
- 可以用如下形式的两个规则来替代：
  
  $$
  S \rightarrow 3sgAux \ 3sgNP \ VP \\
S \rightarrow Non3sgAux \ Non3sgNP \ VP \\
  $$
- 再分别指定第三人称单数和非第三人称单数的助动词形态。这样的方法会导致语法规模增加。

## 动词短语和次范畴化

- 动词短语包括动词和其他一些成分的组合，包括NP和PP以及两者的组合。整个的嵌入句子也可以跟随在动词之后，成为句子补语。
- 动词短语的另一个潜在成分是另一个动词短语。
- 动词后面也可以跟随一个小品词，小品词类似于借此，但与动词组合在一起是构成一个短语动词，与动词不可分割。
- 次范畴化即再分类。传统语法把动词次范畴化为及物动词和不及物动词，而现代语法已经把动词区分为100个次范畴。讨论动词和可能的成分之间的关系是将动词看成一个谓词，而成分想象成这个谓词的论元(argument)。
- 对于动词和它的补语之间的关系，我们可以用上下文无关语法表示一致关系特征，且需要区分动词的各个次类。

## 助动词

- 助动词是动词的一个次类，具有特殊的句法约束。助动词包括情态动词、完成时助动词、进行时助动词、被动式助动词。每一个助动词都给他后面的动词形式一个约束，且需要按照一定的顺序进行结合。
- 四种助动词给VP次范畴化时，VP的中心动词分别是光杆动词、过去分词形式、现在分词形式、过去分词形式。
- 一个句子可以用多个助动词，但是要按照情态助动词、完成时助动词、进行式助动词、被动式助动词的顺序。

## 树图资料库

- 上下文无关语法可以将一个句子剖析成一个句法剖析树，如果一个语料中所有句子都以句法剖析树的形式表示，这样的句法标注了的语料就称为树图资料库(treebank)。
- 树图资料库中的句子隐含的组成了一种语言的语法，我们可以对于每一棵句法剖析树提取其中的CFG规则。从宾州树库中提取出来的CFG规则非常扁平化，使得规则数量很多且规则很长。
- 在树库中搜索需要一种特殊的表达式，能够表示关于节点和连接的约束，用来搜索特定的模式。例如tgrep或者TGrep2。
- 在tgrep、TGrep2中的一个模式由一个关于节点的描述组成，一个节点描述可以用来返回一个以此节点为根的子树。
- 可以使用双斜线对某一类模式命名：
  
  $$
  /NNS?/    NN|NNS
  $$
- Tgrep/Tgrep2模式的好处在于能够描述连接的信息。小于号代表直接支配，远小于符号代表支配，小数点代表线性次序。这种对于连接的描述反应在剖析树中的关系如下：
  ![FoZ2b4.png](https://s2.ax1x.com/2019/01/03/FoZ2b4.png)

## 中心词和中心词查找

- 句法成分能够与一个词法中心词相关联。在一个简单的词法中心词模型中，每一个上下文无关规则与一个中心词相关联，中心词传递给剖析树，因此剖析树中每一个非终止符号都被一个单一单词所标注，这个单一单词就是这个非终止符号的中心词。一个例子如下：
  ![FoZfa9.png](https://s2.ax1x.com/2019/01/03/FoZfa9.png)
- 为了生成这样一棵树，每一个CFG规则都必须扩充来识别一个右手方向的组成成分来作为中心词子女节点。一个节点的中心词词被设置为其子女中心词的中心词。
- 另一种方式是通过一个计算系统来完成中心词查找。在这种方式下是依据树的上下文来寻找指定的句子，从而动态的识别中心词。一旦一个句子被解析出来，树将会被遍历一遍并使用合适的中心词来装饰每一个节点。

## 语法等价与范式

- 语法等价包括两种：强等价，即两个语法生成相同的符号串集合，且他们对于每个句子都指派同样的短语结构；弱等价，即两个语法生成相同的符号串集合，但是不给每个句子指派相同的短语结构。
- 语法都使用一个范式，在范式中每个产生式都使用一个特定的形式。例如一个上下文五官与法是sigma自由的，并且如果他们的每个产生式的形式为A->BC或者是A->a，就说明这个上下文无关语法是符合Chomsky范式的，简称CNF。凡是Chomsky范式的语法都具有二叉树形式。任何上下文无关语法都可以转变成一个弱等价的Chomsky范式语法。
- 使用二叉树形式的剖析树能够产生更小的语法。形如A->A B的规则称为Chomsky并连。

## 有限状态语法和上下文无关语法

- 复杂的语法模型必须表示组成性，因而不适合用有限状态模型来描述语法。
- 当一个非终止符号的展开式中也包含了这个非终止符号时，就会产生语法的递归问题。
- 例如，使用正则表达式来描述以Nominal为中心的名词短语：
  (Det)(Card)(Ord)(Quant)(AP)Nominal(PP)*
- 为了完成这个正则表达式，只需要按顺序展开PP，展开结果为(P NP)*，这样就出现了地柜问题，因为此时出现了NP，在NP的正则表达式中出现了NP。
- 一个上下文无关语法能够被有限自动机生成，当且仅当存在一个生成语言L的没有任何中心自嵌入递归的上下文无关语法。

## 依存语法

- 依存语法与上下文无关语法相对，其句法结构完全由词、词与词之间的语义或句法关系描述。一个例子如下：
  ![FoZOVH.png](https://s2.ax1x.com/2019/01/03/FoZOVH.png)
- 其中没有非终止符号或者短语节点，树中的连接只将两个词语相连。连接即依存关系，代表着语法功能或者一般的语义联系，例如句法主语、直接对象、间接宾语、时间状语等等。
- 依存语法具有很强的预测剖析能力，且在处理具有相对自由词序的语言时表现更好。

# 第十三章：基于上下文无关语法的剖析

## 剖析即搜索

- 在句法剖析中，剖析可以看成对一个句子搜索一切可能的剖析树空间并发现正确的剖析树。
- 对于某一个句子（输入符号串），剖析搜索的目标是发现以初始符号S为根并且恰好覆盖整个输入符号串的一切剖析树。搜索算法的约束来自两方面：
  - 来自数据的约束，即输入句子本身，搜索出来的剖析树的叶子应该是原句的所有单词。
  - 来自语法的约束，搜索出来的剖析树应该有一个根，即初始符号S
- 根据这两种约束，产生了两种搜索策略：自顶向下，目标制导的搜索；自下而上，数据制导的搜索。
- 对于自顶向下的搜索，从根开始，我们通过生成式不断生成下一层的所有可能子节点，搜索每一层的每一种可能，如下图（对于句子book that flight）：
  ![FoZh5R.png](https://s2.ax1x.com/2019/01/03/FoZh5R.png)
- 对于自底向上的搜索，剖析从输入的单词开始，每次都使用语法中的规则，试图从底部的单词向上构造剖析树，如果剖析树成功的构造了以初始符号S为根的树，而且这个树覆盖了整个输入，那么就剖析成功。首先通过词表将每个单词连接到对应的词类，如果一个单词有不止一个词类，就需要考虑所有可能。与自顶向下相反，每次进入下一层时，自底向上需要考虑被剖析的成分是否与某个规则的右手边相匹配，而自顶向下是与左手边相匹配。中途如果无法匹配到规则则将这个树枝从搜索空间中删除，如下图所示：
  ![FoZI8x.png](https://s2.ax1x.com/2019/01/03/FoZI8x.png) 
- 两者对比：
  - 自顶向下是从S开始搜索的，因此不会搜索那些在以S为根的树中找不到位置的子树，而自底向上会产生许多不可能的搜索树
  - 相对应的，自顶向下把搜索浪费在了不可能产生输入单词序列的树上
  - 综上，我们需要将自顶向下和自底向上相结合

## 歧义

- 在句法剖析中需要解决的一个问题是结构歧义，即语法会给一个句子多种剖析结果可能。
- 最常见的两种歧义：附着歧义和并列连接歧义。
- 如果一个特定的成分可以附着在剖析树的一个以上的位置，句子就会出现附着歧义。例如We saw the Eiffel Tower flying to Paris一句中,flying to Paris可以修饰Eiffel Tower也可以修饰We。
- 在并列连接歧义中，存在着不同的短语，这些短语之间用and这样的连接词相连。例如old men and women可以是老年男性和老年女性，或者老年男性和普通女性，即old是否同时分配到men和women上。
- 以上两种歧义还能相互组合嵌套形成更复杂的歧义。假如我们不消歧，仅仅返回所有的可能，留给用户或者人工判断，则随着剖析句子结构变复杂或者剖析规则的增加，得到的可能是成指数级增长的，具体而言，这种剖析句子可能的增长数和算术表达式插入括号问题相同，以Catalan数按指数增长：
  
  $$
  C(n)=\frac{1}{1+n} C_{2n}^n
  $$
- 摆脱这种指数爆炸的方法有两个：
  - 动态规划，研究搜索空间的规律性，使得常见的部分只推导一次，减少与歧义相关的开销
  - 使用试探性的方法来改善剖析器的搜索策略
- 使用例如深度优先搜索或者宽度优先搜索之类的有计划与回溯的搜索算法是在复杂搜索空间中搜索常用的算法，然而在复杂语法空间中无处不在的歧义使得这一类搜索算法效率低下，因为有许多重复的搜索过程。

## 动态规划剖析方法

- 在动态规划中，我们维护一个表，系统的将对于子问题的解填入表中，利用已经存储的子问题的解解决更大的子问题，而不用重复从头开始计算。
- 在剖析中，这样的表用来存储输入中各个部分的子树，当子树被发现时就存入表中，以便以后调用，就这样解决了重复剖析的问题（只需查找子树而不需要重新剖析）和歧义问题（剖析表隐含的存储着所有可能的剖析结果）。
- 主要的三种动态规划剖析方法有三种，CKY算法、Earley算法和表剖析算法。

### CKY剖析

- CKY剖析要求语法必须满足Chomsky范式，即生成式右边要么时两个非终止符号要么是一个终止符号。如果不是Chomsky范式，则需要把一个一般的CFG转换成CNF：
  - 右边有终止符号也有非终止符号：给右边的终止符号单独建一个非终止符号，例如：INF-VP → to VP，改成INF-VP → TO VP和TO → to
  - 右边只有一个非终止符号：这种非终止符号称为单元产物，它们最终会生成非单元产物，用最终生成的非单元产物规则来替换掉单元产物
  - 右边不止2个符号：引入新的非终止符号将规则分解
  - 词法规则保持不变，但是在转换的过程中可能会生成新的词法规则
- 当所有的规则都转换成CNF之后，表中的非终止符号在剖析中有两个子节点，且表中每一个入口代表了输入中的某个区间，对于某个入口例如[0,3]，其可以被拆分成两部分，假如一部分为[0,2]，则另一部分为[2,3]，前者在[0,3]的左边，后者在[0,3]的正下方，如下图：
  ![FoZo26.png](https://s2.ax1x.com/2019/01/03/FoZo26.png)
- 接下来就是如何填表，我们通过自底向上的方法来剖析，对于每个入口[i,j]，包含了输入中i到j这一区间部分的表格单元都会对这个入口值做出贡献，即入口[i,j]左边的单元和下边的单元。下表中的CKY伪算法图描述了这一过程：
  ![FoZjIA.png](https://s2.ax1x.com/2019/01/03/FoZjIA.png)
- 外层循环从左往右循环列，内层循环从下往上循环行，而最里面的循环式遍历串[i,j]的所有可能二分子串，表中存的是可以代表[i,j]区间符号串的非终止符号集合，因为是集合，所以不会出现重复的非终止符号。
- 现在我们完成了识别任务，接下来是剖析。剖析即在[0,N]入口，对应整个句子，找到一个非终止符号作为起始符号S。首先我们要对算法做两点更改：
  - 存入表中的不仅仅是非终止符号，还有其对应的指针，指向生成这个非终止符号的表入口
  - 允许一个入口中存在同一个非终止符号的不同版本
- 做了这些改动之后，这张表就包含了一个给定输入的所有可能剖析信息。我们可以选择[0,N]入口中任意一个非终止符号作为起始符号S，然后根据指针迭代提取出剖析信息。
- 当然，返回所有的可能剖析会遇到指数爆炸的问题，因此我们在完整的表上应用维特比算法，计算概率最大的剖析并返回这个剖析结果。

### Early算法

- 相比CKY自底向上的剖析，Early算法采用了自顶向下的剖析，而且只用了一维的表保存状态，每个状态包含三类信息：
  - 对应某一单一语法规则的子树
  - 子树的完成状态
  - 子树对应于输入中的位置
- 算法流程图如下：
  ![FoZHKO.png](https://s2.ax1x.com/2019/01/03/FoZHKO.png)
- 算法对于状态的操作有三种：
  - 预测：造出一个新的状态来表示在剖析过程中生成的自顶向下的预测。当待剖析的状态为非终极符号但又不是词类范畴时，对于这个非终极符号的不同展开，预测操作都造出一个新的状态。
  - 扫描：当待剖析的状态是词类范畴时，就检查输入符号串，并把对应于所预测的词类范畴的状态加入线图中。
  - 完成：当右边所有状态剖析完成时，完成操作查找输入中在这个位置的语法范畴，发现并推进前面造出的所有状态。

### 表剖析

- 表剖析允许动态的决定表格处理的顺序，算法动态的依照计划依次删除图中的一条边，而计划中的元素排序是由规则决定的。
  ![FoZTxK.png](https://s2.ax1x.com/2019/01/03/FoZTxK.png)

## 部分剖析

- 有时我们只需要输入句子的部分剖析信息
- 可以用有限状态自动机级联的方式完成部分剖析，这样会产生比之前提到的方法更加“平”的剖析树。
- 另一种有效的部分剖析的方法是分块。使用最广泛覆盖的语法给句子做词类标注，将其分为有主要词类标注信息且不没有递归结构的子块，子块之间不重叠，就是分块。
- 我们用中括号将每一个分块框起来，有可能一些词并没有被框住，属于分块之外。
- 分块中最重要的是基本分块中不能递归包含相同类型的成分。

### 基于规则的有限状态分块

- 利用有限状态方式分块，需要为了特定目的手动构造规则，之后从左到右，找到最长匹配分块，并接着依次分块下去。这是一个贪心的分块过程，不保证全局最优解。
- 这些分块规则的主要限制是不能包含递归。
- 使用有限状态分块的优点在于可以利用之前转录机的输出作为输入来组成级联，在部分剖析中，这种方法能够有效近似真正的上下文无关剖析器。

### 基于机器学习的分块

- 分块可以看成序列分类任务，每个位置分类为1（分块）或者0（不分块）。用于训练序列分类器的机器学习方法都能应用于分块中。
- 一种卓有成效的方法是将分块看成类似于词类标注的序列标注任务，用一个小的标注符号集同时编码分块信息和每一个块的标注信息，这种方式称为IOB标注，用B表示分块开始，I表示块内，O表示块外。其中B和I接了后缀，代表该块的句法信息。
- 机器学习需要训练数据，而分块的已标数据很难获得，一种方法是使用已有的树图资料库，例如宾州树库。

### 评价分块系统

- 准确率：模型给出的正确分块数/模型给出的总分块数
- 召回率：模型给出的正确分块数/文本中总的正确分块数
- F1值：准确率和召回率的调和平均

# 第十四章：统计剖析

## 概率上下文无关语法

- 概率上下文无关语法PCFG是上下文无关语法的一种简单扩展，又称随机上下文无关语法。PCFG在定义上做出了一点改变：
  - N：非终止符号集合
  - Σ：终止符号集合
  - R：规则集合，与上下文无关语法相同，只不过多了一个概率p，代表某一项规则执行的条件概率$P(\beta|A)$
  - S：一个指定的开始符号
- 当某个语言中所有句子的概率和为1时，我们称这个PCFG时一致的。一些递归规则可能导致PCFG不一致。

## 用于消歧的PCFG

- 对于一个给定句子，其某一特定剖析的概率是所有规则概率的乘积，这个乘积既是一个剖析的概率，也是剖析和句子的联合概率。这样，对于出现剖析歧义的句子，其不同剖析的概率不同，通过选择概率大的剖析可以消歧。

## 用于语言建模的PCFG

- PCFG为一个句子分配了一个概率（即剖析的概率），因此可以用于语言建模。相比n元语法模型，PCFG在计算生成每一个词的条件概率时考虑了整个句子，效果更好。对于含歧义的句子，其概率是所有可能剖析的概率之和。

## PCFG的概率CKY剖析

- PCFG的概率剖析问题：为一个句子产生概率最大的剖析
- 概率CKY算法扩展了CKY算法，CKY剖析树中的每一个部分被编码进一个$(n+1)*(n+1)$的矩阵（只用上三角部分），矩阵中每一个元素包含一个非终止符号集合上的概率分布，可以看成每一个元素也是V维，因此整个存储空间为$(n+1)*(n+1)*V$，其中[i,j,A]代表非终止符号A可以用来表示句子的i位置到j位置这一段的概率。
- 算法伪代码：
  ![FoZbrD.png](https://s2.ax1x.com/2019/01/03/FoZbrD.png)
- 可以看到也是用k对某一区间[i,j]做分割遍历，取最大的概率组合作为该区间的概率，并向右扩展区间进行动态规划。

## 学习到PCFG的规则概率

- 上面的伪算法图用到了每一个规则的概率。如何获取这个概率？两种方法，第一种朴素的方法是在一个已知的树库数据集上用古典概型统计出概率：
  
  $$
  P(\alpha \rightarrow \beta | \alpha) = \frac{Count(\alpha \rightarrow \beta)}{\sum _{\gamma} Count(\alpha \rightarrow \gamma)}
  $$
- 假如我们没有树库，则可以用非概率剖析算法来剖析一个数据集，再统计出概率。但是非概率剖析算法在剖析歧义句子时，需要对每一种可能剖析计算概率，但是计算概率需要概率剖析算法，这样就陷入了鸡生蛋蛋生鸡的死循环。一种解决方案是先用等概率的剖析算法，剖析句子，得出每一种剖析得概率，然后用概率加权统计量，然后重新估计剖析规则的概率，继续剖析，反复迭代直到收敛。这种算法称为inside-outside算法，是前向后向算法的扩展，同样也是EM算法的一种特例。

## PCFG的问题

- 独立性假设导致不能很好的建模剖析树的结构性依存：每个PCFG规则被假定为与其他规则独立，例如，统计结果表明代词比名词更有可能称为主语，因此当NP被展开时，如果NP是主语，则展开为代词的可能性较高——这里需要考虑NP在句子种的位置，然而这种概率依存关系是PCFG所不允许的，
- 缺乏对特定单词的敏感，导致次范畴化歧义、介词附着、联合结构歧义的问题：例如在介词附着问题中，某一个介词短语into Afghanistan附着于哪一个部分，在PCFG中计算时被抽象化为介词短语应该附着一个哪一个部分，而抽象化的概率来自于对语料的统计，这种统计不会考虑特定的单词。又例如联合结构歧义，假如一个句子的两种可能剖析树使用了相同的规则，而规则在树中的位置不同，则PCFG对两种剖析计算出相同的概率：因为PCFG假定规则之间是独立的，联合概率是各个概率的乘积。

## 通过拆分和合并非终止符号来改进PCFG

- 先解决结构性依存的问题。之前提到了我们希望NP作为主语和宾语时有不同概率的规则，一种想法就是将NP拆分成主语NP和宾语NP。实现这种拆分的方法是父节点标注，及每个节点标注了其父节点，对于主语NP其父节点是S，对于宾语NP，其父节点是VP，因此不同的NP就得到了区分。除此之外，还可以通过词性拆分的方式增强剖析树。
- 拆分会导致规则增多，用来训练每一条规则的数据变少，引起过拟合。因此要通过一个手写规则或者自动算法来根据每个训练集合并一些拆分。

## 概率词汇化的CFG

- 概率CKY剖析更改了语法规则，而概率词汇化模型更改了概率模型本身。对于每一条规则，不仅要产生成分的规则变化，还要在每个成分上标注其中心词和词性，如下图：
  ![FoeSRP.png](https://s2.ax1x.com/2019/01/03/FoeSRP.png)
- 为了产生这样的剖析树，每一条PCFG规则右侧需要选择一个成分作为中心词子节点，用子节点的中心词和词性作为该节点的中心词和词性。
  其中，规则被分成了两类，内部规则和词法规则，后者是确定的，前者是需要我们估计的：
  ![FoZqqe.png](https://s2.ax1x.com/2019/01/03/FoZqqe.png)
- 我们可以用类似父节点标注的思想来拆分规则，拆分后每一部分都对应一种可能的中心词选择。假如我们将概率词汇话的CFG看成一个大的有很多规则CFG，则可以用之前的古典概型来估计概率。但是这样的效果不会很好，因为这样的规则划分太细了，没有足够的数据来估计概率。因此我们需要做出一些独立性假设，将概率分解为更小的概率乘积，这些更小的概率能容易从语料中估计出来。
- 不同的统计剖析器区别在于做出怎样的独立性假设。
- Collins剖析如下图所示：
  ![FoZzGt.png](https://s2.ax1x.com/2019/01/03/FoZzGt.png)
- 其概率拆解为：
  
  $$
  P(VP(dumped,VBD)→VBD(dumped,VBD)NP(sacks,NNS)PP(into,P))= \\
P_H (VBD│VP,dumped)* \\
P_L (STOP│VP,VBD,dumped)* \\
P_R (NP(sacks,NNS)│VP,VBD,dumped)* \\
P_R (PP(into,P)│VP,VBD,dumped)* \\
P_R (STOP|VP,VBD,dumped) \\
  $$
- 给出生成式左边之后，首先生成规则的中心词，之后一个一个从里到外生成中心词的依赖。先从中心词左侧一直生成直到遇到STOP符号，之后生成右边。如上式做出概率拆分之后，每一个概率都很容易从较小的数据量中统计出来。完整的Collins剖析器更为复杂，还考虑了词的距离关系、平滑技术、未知词等等。

## 评价剖析器

- 剖析器评价的标准方法叫做PARSEVAL测度，对于每一个句子s：
  - 标记召回率=(Count(s的候选剖析中正确成分数）)/(Count(s的树库中正确成分数）)
  - 标记准确率=(Count(s的候选剖析中正确成分数）)/(Count(s的候选剖析中全部成分数）)

## 判别式重排序

- PCFG剖析和Collins词法剖析都属于生成式剖析器。生成式模型的缺点在于很难引入任意信息，即很难加入对某一个PCFG规则局部不相关的特征。例如剖析树倾向于右生成这一特征就不方便加入生成式模型当中。
- 对于句法剖析，有两类判别式模型，基于动态规划的和基于判别式重排序的。
- 判别式重排包含两个阶段，第一个阶段我们用一般的统计剖析器产生前N个最可能的剖析及其对应的概率序列。第二个阶段我们引入一个分类器，将一系列句子以及每个句子的前N个剖析-概率对作为输入，抽取一些特征的大集合并针对每一个句子选择最好的剖析。特征包括：剖析概率、剖析树中的CFG规则、平行并列结构的数量、每个成分的大小、树右生成的程度、相邻非终止符号的二元语法、树的不同部分出现的次数等等。

## 基于剖析的语言建模

- 使用统计剖析器来进行语言建模的最简单方式就是利用之前提到的二阶段算法。第一阶段我们运行一个普通的语音识别解码器或者机器翻译解码器（基于普通的N元语法），产生N个最好的候选；第二阶段，我们运行统计剖析器并为每一个候选句分配一个概率，选择概率最佳的。

## 人类剖析

- 人类在识别句子时也用到了类似的概率剖析思想，两个例子：
  - 对于出现频率高的二元语法，人们阅读这个二元语法所花的时间就更少
  - 一些实验表明人类在消歧时倾向于选择统计概率大的剖析

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
        data-lang="zh-CN"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>