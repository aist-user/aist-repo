from rouge_score import rouge_scorer


def precision(result, gold):
    precision_result = {}
    for key, values in result.items():
        splitted = key.split('_')
        topic, title = int(splitted[1]), splitted[0]
        summary = values['summary_list']
        gold_item = gold[title]
        count_right = 0
        for headline in gold_item.values():
            if type(headline['topic_num']) is list:
                t = headline['topic_num'][0]
            else:
                t = headline['topic_num']
            if t == topic:
                text = ''.join(headline['text']).replace('\n', '')
                for s in summary:
                    if s.strip() in text.strip():
                        count_right += 1
                count_right /= 3
        precision_result[key] = count_right
    return precision_result


def rouge(result, gold):
    f1 = []
    recall = []
    precisions = []
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    count = 0
    for key, values in result.items():
        splitted = key.split('_')
        topic, title = int(splitted[1]), splitted[0]
        summary = values['summary_list']
        gold_item = gold[title]
        for headline in gold_item.values():
            if type(headline['topic_num']) is list:
                t = headline['topic_num'][0]
            else:
                t = headline['topic_num']
            if t == topic:
                text = ''.join(headline['text']).replace('\n', '')
                score = scorer.score(text, ' '.join(summary))
                f1.append(score['rouge2'].fmeasure)
                recall.append(score['rouge2'].recall)
                precisions.append(score['rouge2'].precision)
        count += 1
        #print(count)
    return f1, precisions, recall
