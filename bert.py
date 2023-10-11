from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# 下游文本纠错任务的全连接层模型
class TextCorrectionModel(nn.Module):
    def __init__(self, bert_model):
        super(TextCorrectionModel, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(bert_model.config.hidden_size, bert_model.config.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state

        # 全连接层和激活函数处理
        fc_output = self.activation(self.fc(hidden_states))

        # 输出层处理
        corrected_output = self.output_layer(fc_output)

        return corrected_output

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')
model = BertModel.from_pretrained('bert_model')

# 输入文本
text = ["我爱自然语言处理，也热爱人工智能。", "我爱人工智能"]

# 使用jieba进行分词
#seg_list = jieba.cut(text)
#word_tokens = list(seg_list)

xx = tokenizer.batch_encode_plus(text, max_length=10, padding='max_length', truncation=True)
s = xx.input_ids
xxx = model(input_ids=xx.input_ids, attention_mask=xx.attention_mask).last_hidden_state

# 分词和转换为模型输入格式
tokens = tokenizer(text, return_tensors="pt")

# 调用下游文本纠错任务模型
text_correction_model = TextCorrectionModel(model)
output = text_correction_model(tokens.input_ids, tokens.attention_mask, tokens.token_type_ids)

# 获取预测标签
predicted_labels = torch.argmax(output, dim=2).squeeze().tolist()

# 将分数最高的token转换为汉字
predicted_indexes = torch.argmax(output, dim=2)
predicted_tokens = [tokenizer.decode(idx) for idx in predicted_indexes.tolist()[0]]
predicted_text = "".join(predicted_tokens)

# 将预测标签转换为实体序列
#entity_labels = [tokenizer.convert_ids_to_tokens(label_id) for label_id in predicted_labels]
#entity_labels = entity_labels[1:-1]  # 去掉[CLS]和[SEP]标记
# 将预测标签转换为实体序列
entity_labels = [tokenizer.convert_ids_to_tokens(label_id) for label_id in predicted_labels]
entity_labels = [label for sublist in entity_labels for label in sublist]  # 展开列表


# 打印实体序列
print("实体序列:", entity_labels)
