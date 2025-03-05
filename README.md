# Hip

![Methods](https://github.com/Richardqiyi/Hip/blob/main/Methods.png)

### Experimental Results

| Evaluation Metric | FeatureFusion | LearnedFeatureFusion | Non-image only |
|------------------|----------|----------|----------|
| Accuracy         | 0.7049   | 0.7705   | 0.7541   |
| Precision        | 0.2857   | 0.3333   | 0.3846   |
| Recall           | 0.3333   | 0.1667   | 0.4167   |
| F1 Score         | 0.3077   | 0.2222   | 0.4000   |
| Specificity      | 0.7959   | 0.9184   |          |
| AUC-ROC          | 0.6463   | 0.7381   |          |
| AUC-PR           | 0.2481   | 0.3214   |          |

### Usage

```
python train.py --train_img_path <path_to_your_train-dataset> \
                --validation_img_path <path_to_your_validation-dataset> \
                --test_img_path <path_to_your_test-dataset> \
                --out_dir <path_to_results> \
                --model learned-feature-fusion <or feature-fusion> \
                --fusion_mode concat \
                --lr <learning rate> \
                --label_smoothing 0.01
```
