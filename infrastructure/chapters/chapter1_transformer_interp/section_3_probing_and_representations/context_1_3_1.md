# Linear Probes

Refusal in LLMs is mediated by a single direction 

### Josh Engels
Hm off the top of my head possibly this paper is good to reproduce?
https://arxiv.org/pdf/2406.07882
There’s also all of the datasets they studied in the representation engineering paper, our paper on SAE probing https://arxiv.org/pdf/2502.16681, Wes’s paper on sparse probing https://arxiv.org/abs/2305.01610, Wes’s space and time paper https://arxiv.org/abs/2310.02207, apollos recent deception probing paper
This paper is also skeptical of probing being differentials useful and I think is great https://arxiv.org/abs/2312.03729
Exercise wise definitely training probes with hyperparameters fitting using a Val set with a held out test set, possibly testing on ood data and quantifying how well the probe generalizes, proving on different layers / on different components
Not sure I know of good people to design it but I’ll let you know if I think of any. I could theoretically help out some but likely not until late may
I also think the difference between probing on model generations vs text that the model didn’t generate is an important distinction

### Connor K
- Apollo deception probes good
- Also coo probes from Redwood
- LW "how do truth probes generalize"
- Lastly Sam Marks' geometry of truth

### Notes from GDM folk
Useful reference material internally?
- Sen's "Probe training.ipynb", Jun 16 2023, training SVM probes to separate factual knowledge by category. URL here  
- Lewis' "probe_deep_dive.ipynb", Dec 18th 2024, detecting harmful intent + jailbreaking behaviour, also compares to SAEs. URL here  
These are a bit out of date though, I think

### Other random shit
https://docs.google.com/document/d/11n6BePT2hDlp8AZ6nyhaVULuRqyvxW-lCEflkds5tj8/edit?tab=t.0

# Exercises in progress
https://chatgpt.com/canvas/shared/688a2a6471148191bd8de31207a928a4




---

[This](https://arxiv.org/abs/2310.02207) should be in linear representations: Language Models Represent Space and Time

Sam Marks' thoughts on GDM's deception & probing stuff: https://xcancel.com/saprmarks/status/1997117843083567235 

Also general, find a place for https://www.alignmentforum.org/posts/DqaoPNqhQhwBFqWue/principles-for-picking-practical-interpretability-projects
