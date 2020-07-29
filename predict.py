import classifier
import testing.intent_predictor as pred

import eval.metrics as metrics

model_path = '/mount/arbeitsdaten/studenten1/team-lab-nlp/mandavsi_rileyic/saved_models/experiment_4'
y_pred, y_true = pred.load_model_and_predict_test_data(model_path)

confusion_matrix = metrics.get_confusion_matrix(y_true, y_pred)

print(confusion_matrix)

metrics.plot_confusion_matrix(confusion_matrix, "BiLSTM Classifier + Attention with ELMo", model_path+'/confusion_matrix_plot.png')
