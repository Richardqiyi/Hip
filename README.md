# Hip


### Experimental Results

| Evaluation Metric | zero-shot | post-training|ViT with metadata|CLIP mutimodal|
|------------------|----------|----------|--------|--------|
| Accuracy         | 0.7213  | 0.7541    | 0.7541 |0.7541  |
| Precision        | 0.2222  | 0.3333   | 0.2857  |0.2857  |
| Recall           | 0.1667   | 0.2500  | 0.1667  |0.1667  |
| F1 Score         | 0.1905  | 0.2857   | 0.2105  |0.2105  |
| Specificity      | 0.8571   | 0.8776   | 0.8980 | 0.8980 |
| AUC-ROC          |         | 0.6259   | 0.6854  |0.6854  |
| AUC-PR           |        | 0.3127   | 0.2700   |0.2700  |

### Zero-shot Prompt

```
positive_prompt = "during follow-up the patient converted to a hip prosthesis"
negative_prompt = "the patient still has the native hip"
```
