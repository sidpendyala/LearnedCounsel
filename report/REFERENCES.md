# Verified references (checked online 2026-11-25)

Cite only what is listed here. "Supports" says what the source was checked for.

1. **LegalBench.** Guha, N., et al. (2023). *LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models.* Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track. arXiv:2308.11462. https://arxiv.org/abs/2308.11462
   - Supports: the 16 `learned_hands_*` tasks (from the task READMEs, github.com/HazyResearch/legalbench), binary and class balanced, "negatives" randomly sampled from posts with other topics, license CC BY-NC-SA 4.0, the category definitions.
2. **Learned Hands.** Stanford Legal Design Lab and Suffolk LIT Lab. *Learned Hands* (project and labeled dataset). https://learnedhands.law.stanford.edu and https://spot.suffolklitlab.org/data/#learnedhands
   - Supports: labeling game started in autumn 2018, legal stories from Reddit's legal advice board, labels by people with legal expertise, public dataset. No formal paper or DOI was found: cite as a project with its URL.
3. **LIST.** Stanford Legal Design Lab. *Legal Issues Taxonomy (LIST).* https://betterinternet.law.stanford.edu/about-the-project/taxonomy-nsmiv2
   - Supports: LIST is the codebook behind the Learned Hands labels, with roots in the NSMI taxonomy. (The article "Every legal problem that exists" is mentioned there, but its authors, year and venue were not verified: do not cite it.)
4. **Justice Gap.** Legal Services Corporation (2022). *The Justice Gap: The Unmet Civil Legal Needs of Low-income Americans.* Conducted with NORC at the University of Chicago. https://justicegap.lsc.gov
   - Supports: low-income Americans got no or not enough legal help for 92% of the civil legal problems that **substantially affected** them (2021 survey). It is not 92% of all problems: keep the qualifier. (Exact title still to be copied from the report's cover page.)
5. **Multi-agent deliberation in law.** Steging, C., van Leeuwen, L., & Zbiegień, T. (2026). *Investigating Multi-Agent Deliberation in Law.* arXiv:2606.30906. Accepted at the AIDA2J Workshop, 21st International Conference on AI and Law (ICAIL), Singapore, June 8, 2026. https://arxiv.org/abs/2606.30906
   - Supports: multi-agent frameworks "achieve comparable overall performance to baseline large language models" but "produce significantly distinct answers", and each side solves cases the other does not. Body text: five yes/no benchmarks, 1,250 questions, GPT-5-mini; average F1 scores "differ by no more than 1.5 points". The per-dataset scores and the conclusion section were not read.
6. **Model.** Yang, A., et al. (Qwen Team) (2025). *Qwen3 Technical Report.* arXiv:2505.09388. https://arxiv.org/abs/2505.09388
   - Supports: Qwen3 model family, thinking and non-thinking modes, Apache 2.0.
7. **Self-consistency (B5).** Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. arXiv:2203.11171. https://arxiv.org/abs/2203.11171
   - Supports: sampling several answers and taking the most consistent one. Our B5 votes on labels from direct answers, not on chain-of-thought reasoning paths, so it is an adaptation.
8. **Software.** Ollama 0.40.17 (https://ollama.com), scikit-learn, SciPy, pandas, pydantic, Streamlit, matplotlib (versions in requirements.txt).

Standard methods named without a citation so far (McNemar's test, Holm's correction, the bootstrap): verify a source before adding one.
