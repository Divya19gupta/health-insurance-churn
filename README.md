# Causal Analysis of Member Switching in German Health Insurance

This repository contains the code, data pipeline, and results for the study **"Causal Analysis of Member Switching in German Health Insurance"**, which combines predictive modeling and causal inference to understand how additional contributions (Zusatzbeiträge) affect member churn in Germany’s statutory health insurance (SHI) system from 2016 to 2025.

---

## 📌 Overview

This project investigates how changes in contribution rates affect member switching behavior using both Random Forest (prediction) and DoWhy (causal inference).

### Research Questions:
- **RQ1:** What is the causal impact of a fund’s additional contribution increase on its own churn rate?
- **RQ2:** How do competitor contribution rates influence switching?
- **RQ3:** How do causal inference methods compare to predictive models in explaining churn?

---

## 📊 Methods

### 🔍 Predictive Modeling
- **Model:** Random Forest Classifier
- **Features:**
  - Fund’s own contribution rate (`avg_zusatzbeitrag`)
  - Competitor rates
  - Market share (`marktanteil_mitglieder`)
  - Risk adjustment factor (`risikofaktor`)
  - Encoded fund name and survey responses
- **Output:** Binary classification (Churn: Yes/No)
- **Performance:** ~81% accuracy

### 🧠 Causal Inference (DoWhy)
- **Model A:** Effect of a fund’s own contribution increase
- **Model B:** Effect of competitors’ contribution rate
- **Method:** Backdoor adjustment using causal graphs
- **Validation:** Placebo and refutation tests for robustness

---

## 📈 Results

- **Predictive model** showed that market share and fund identity are strong churn predictors.
- **Causal analysis** revealed:
  - A statistically significant increase in churn after a fund raised its own rate.
  - Competitor rates also influence switching behavior.

---

## 💡 Key Takeaways

- Predictive models explain **who** is likely to churn.
- Causal models explain **why** members churn — especially in response to pricing.
- Causal methods provide policy-relevant insights, especially in regulated markets.

---
## 📖 References

1. Lötsch, J., & Mayer, B. (2022). A biomedical case study showing that tuning random forests can fundamentally change the interpretation of supervised data structure exploration aimed at knowledge discovery. *BioMedInformatics, 2*(4), 544–552. https://doi.org/10.3390/biomedinformatics2040034  

2. Sam, G., Asuquo, P., & Stephen, B. (2024). Customer churn prediction using machine learning models. *Journal of Engineering Research and Reports, 26*(2), 181–193. https://doi.org/10.9734/jerr/2024/v26i21081  

3. OpenAI. (2025). ChatGPT (May 30 version) [Large language model]. https://chat.openai.com/  

4. Grammarly Inc. (n.d.). Grammarly [AI writing assistant]. https://www.grammarly.com/  

5. QuillBot. (n.d.). QuillBot [AI paraphrasing tool]. https://quillbot.com/  

6. Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324  

7. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. https://www.cambridge.org/9780521895606  

8. Sharma, A., & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference. *arXiv preprint* arXiv:2011.04216. https://arxiv.org/abs/2011.04216  

9. Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference: Foundations and Learning Algorithms*. MIT Press. https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/  
https://webdav.tuebingen.mpg.de/causeme/ElementsCausalInference.pdf  

10. Microsoft. (2021). EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation. https://github.com/microsoft/EconML  

11. Shalit, U., Johansson, F. D., & Sontag, D. (2017). Estimating individual treatment effect: Generalization bounds and algorithms. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 3076–3085). https://proceedings.mlr.press/v70/shalit17a.html


