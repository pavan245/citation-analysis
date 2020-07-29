import classifier
import testing.intent_predictor as pred

import eval.metrics as metrics

y_pred, y_true = pred.load_model_and_predict_test_data("/mount/arbeitsdaten/studenten1/team-lab-nlp/mandavsi_rileyic/saved_models/experiment_4")

confusion_matrix = metrics.get_confusion_matrix(y_true, y_pred)

print(confusion_matrix)

metrics.plot_confusion_matrix(confusion_matrix, "BiLSTM Classifier + Attention with ELMo")
