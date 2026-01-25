## Exercise Set 4: Interpreting Reasoning (Thought Anchors)  
There are some kinda basic parsing errors in the dataset. For example, in the case study example they have, for forced completions, they extract the answer as "20" in one of the examples and classify it as "False" because the correct answer is 19, but in reality it should have been classified as True because here was the full rollout, showing the model did converge on 19:

```
'Therefore, the final answers is \\boxed{20}. However, upon ' 're-examining, the correct number of bits should be 19 because ' 'the value in decimal is less than 2¹⁹, requiring 19 bits. So, ' 'the correct answer is \\boxed{19}.',
```

Current task list for this one? I've got to fill in the dropdown "Click here to see what you should find", although I'm not really sure how to interpret some of these.

Then, add "make your own resampling method" as a bonus exercise in this section (including KL div!), and move on to planning the whitebox section.
	...or maybe KL div can be an exercise before this point, because I actually can look at the KL div by just doing a model fwd pass?!

...

In most of this chapter's content so far, we've focused on dividing up LLM computation into small steps, specifically single-token generation. We saw this in Indirect Object Identification where we deeply analyzed how GPT2-Small generates a single token, and we can also see it in other exercises like OthelloGPT. But recent models have made advances in **chain-of-thought reasoning**. This introduces a new dynamic to model capabilities: rather than just thinking about serialized computation over layers to produce a single token, we now have to think about serialized computation over a very large number of tokens. To get traction on this problem, we're going to need to introduce a new abstraction.