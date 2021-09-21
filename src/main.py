import json

import numpy as np

import metrics
from AbaeLexRank import AbaeLexRank
from ArtmLexRank import BiasedLexRankArtm
from data_reader import DataReader
from rus_preprocessing_udpipe import UdpipeProcessor


def start_artm(d_value):
    reader = DataReader()
    data, plain_texts, topic_idx = reader.read_json_to_flat('artm/min_person_all.json')
    topics = reader.get_json('artm/topics_artm.json')
    udpipe_proc = UdpipeProcessor()

    for k, v in data.items():
        for head, value in data[k].items():
            s = ' '.join(data[k][head]['text']).replace('\n', '').strip().strip('.').split('.')
            data[k][head]['len'] = len(s)

    processed = 0
    failed_topic = 0
    s_count = 3
    d = d_value
    summaries = {}

    summarizer = BiasedLexRankArtm()

    for key, text in plain_texts.items():
        is_ok = False
        sentences = udpipe_proc.tag_ud(text.strip('.'))
        if any([data[key][headline]['len'] <= s_count for headline in data[key].keys()]):  # or len(topic_idx[key]) < 2
            continue

        for t in topic_idx[key]:
            topic = t
            if type(t) is list:
                topic = t[0]
            if topic > 49:
                failed_topic += 1
                continue
            topic_bias = ' '.join(topics[f'topic_{topic}'])
            topic_bias = udpipe_proc.tag_ud(topic_bias)

            idx = summarizer(sentences, topic_bias, d=d, sentences_count=s_count)
            s = text.replace('\n', '').strip('.').split('.')
            is_ok = True
            try:
                summaries[f'{key}_{topic}'] = {'bias': topic_bias, 'summary_list': [s[i] for i in idx]}
            except:
                print('failure')

        processed += is_ok
        print(processed)

    print(processed)

    with open(f"results/artm_summaries_s{s_count}_d{d}.json", "w", encoding='utf-8') as outfile:
        json.dump(summaries, outfile, ensure_ascii=False)


def start_abae(d_value, temp):
    reader = DataReader()
    data, plain_texts, topic_idx = reader.read_json_to_flat('abae/embeddings_wiki_abae_spacy_1.json')
    #topics = reader.get_json('abae/emb_aspects_50.json')

    for k, v in data.items():
        for head, value in data[k].items():
            s = ''.join(data[k][head]['text']).replace('\n', '').strip().strip('.').split('.')
            data[k][head]['len'] = len(s)

    processed = 0
    failed_topic = 0
    s_count = 3
    d = d_value
    summaries = {}

    summarizer = AbaeLexRank(t=temp)

    for key, text in plain_texts.items():
        is_ok = False
        sentences = text.split('.')
        sentences = [s for s in sentences if s not in ['', ' ']]

        if any([data[key][headline]['len'] <= s_count for headline in data[key].keys()]):  # or len(topic_idx[key]) < 2
            continue

        for t in topic_idx[key]:
            topic = t
            if type(t) is list:
                topic = t[0]
            if topic > 49:
                failed_topic += 1
                continue

            idx = summarizer(sentences, topic, d=d, sentences_count=s_count)
            s = text.replace('\n', '').strip('.').split('.')
            is_ok = True
            try:
                summaries[f'{key}_{topic}'] = {'topic': topic, 'summary_list': [s[i] for i in idx]}
            except:
                print('failure')

        processed += is_ok
        print(processed)

    print(processed)

    with open(f"results/abae_summaries_s{s_count}_d{d}_t{temp}.json", "w", encoding='utf-8') as outfile:
        json.dump(summaries, outfile, ensure_ascii=False)


def estimate(d, temp):
    with open(f"results/abae_summaries_s3_d{d}.json", encoding='utf-8') as json_file:
        summaries = json.load(json_file)
        with open('abae/embeddings_wiki_abae_spacy_1.json', encoding='utf-8') as json_file:
        #with open('artm/min_person_all.json', encoding='utf-8') as json_file:
            gold_summaries = json.load(json_file)
            print(np.array([i for i in metrics.precision(summaries, gold_summaries).values()]).mean())

            f1, pr, rec = metrics.rouge(summaries, gold_summaries)
            print(np.array(f1).mean())
            print(np.array(pr).mean())
            print(np.array(rec).mean())


for d in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    start_abae(d, 1)
    print(f'd = {d}:')
    estimate(d, 1)
