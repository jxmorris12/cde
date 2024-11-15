---
model-index:
- name: no_model_name_available
  results:
  - dataset:
      config: en
      name: MTEB AmazonCounterfactualClassification (en)
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
      split: test
      type: mteb/amazon_counterfactual
    metrics:
    - type: accuracy
      value: 90.68656716417908
    - type: ap
      value: 65.33121840211768
    - type: ap_weighted
      value: 65.33121840211768
    - type: f1
      value: 86.37167872349994
    - type: f1_weighted
      value: 91.03529441251852
    - type: main_score
      value: 90.68656716417908
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB AmazonPolarityClassification (default)
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
      split: test
      type: mteb/amazon_polarity
    metrics:
    - type: accuracy
      value: 95.72682499999999
    - type: ap
      value: 93.42169887240918
    - type: ap_weighted
      value: 93.42169887240918
    - type: f1
      value: 95.72459795953115
    - type: f1_weighted
      value: 95.72459795953115
    - type: main_score
      value: 95.72682499999999
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB AmazonReviewsClassification (en)
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
      split: test
      type: mteb/amazon_reviews_multi
    metrics:
    - type: accuracy
      value: 56.413999999999994
    - type: f1
      value: 55.56652931258766
    - type: f1_weighted
      value: 55.56652931258766
    - type: main_score
      value: 56.413999999999994
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB ArguAna (default)
      revision: c22ab2a51041ffd869aaddef7af8d8215647e41a
      split: test
      type: mteb/arguana
    metrics:
    - type: main_score
      value: 75.285
    - type: map_at_1
      value: 53.129000000000005
    - type: map_at_10
      value: 68.395
    - type: map_at_100
      value: 68.592
    - type: map_at_1000
      value: 68.593
    - type: map_at_20
      value: 68.57
    - type: map_at_3
      value: 65.351
    - type: map_at_5
      value: 67.41
    - type: mrr_at_1
      value: 54.97866287339972
    - type: mrr_at_10
      value: 69.0639041296936
    - type: mrr_at_100
      value: 69.26100847485964
    - type: mrr_at_1000
      value: 69.26169235712246
    - type: mrr_at_20
      value: 69.23874844194219
    - type: mrr_at_3
      value: 65.990990990991
    - type: mrr_at_5
      value: 68.08558558558559
    - type: nauc_map_at_1000_diff1
      value: 18.4460277770279
    - type: nauc_map_at_1000_max
      value: -20.731930259414142
    - type: nauc_map_at_1000_std
      value: -31.798243642316653
    - type: nauc_map_at_100_diff1
      value: 18.447072434924202
    - type: nauc_map_at_100_max
      value: -20.730332266221783
    - type: nauc_map_at_100_std
      value: -31.80225872296014
    - type: nauc_map_at_10_diff1
      value: 18.379806037885196
    - type: nauc_map_at_10_max
      value: -20.518503464460032
    - type: nauc_map_at_10_std
      value: -31.80765555528529
    - type: nauc_map_at_1_diff1
      value: 20.17612594754981
    - type: nauc_map_at_1_max
      value: -23.058659883369618
    - type: nauc_map_at_1_std
      value: -31.028464230728563
    - type: nauc_map_at_20_diff1
      value: 18.42995778554288
    - type: nauc_map_at_20_max
      value: -20.697263989926775
    - type: nauc_map_at_20_std
      value: -31.827895770005117
    - type: nauc_map_at_3_diff1
      value: 18.36054135768821
    - type: nauc_map_at_3_max
      value: -20.368069921997837
    - type: nauc_map_at_3_std
      value: -31.913305579174295
    - type: nauc_map_at_5_diff1
      value: 18.303234178687582
    - type: nauc_map_at_5_max
      value: -20.1985671806448
    - type: nauc_map_at_5_std
      value: -31.71405391999761
    - type: nauc_mrr_at_1000_diff1
      value: 12.111287924360353
    - type: nauc_mrr_at_1000_max
      value: -23.676405560467163
    - type: nauc_mrr_at_1000_std
      value: -31.006243370938325
    - type: nauc_mrr_at_100_diff1
      value: 12.112518888407333
    - type: nauc_mrr_at_100_max
      value: -23.674714351936615
    - type: nauc_mrr_at_100_std
      value: -31.010304852688403
    - type: nauc_mrr_at_10_diff1
      value: 12.099815722180036
    - type: nauc_mrr_at_10_max
      value: -23.433136661265568
    - type: nauc_mrr_at_10_std
      value: -31.02114709545547
    - type: nauc_mrr_at_1_diff1
      value: 15.016761063587952
    - type: nauc_mrr_at_1_max
      value: -24.180527818542174
    - type: nauc_mrr_at_1_std
      value: -29.245184988869717
    - type: nauc_mrr_at_20_diff1
      value: 12.101158823054263
    - type: nauc_mrr_at_20_max
      value: -23.638727453942774
    - type: nauc_mrr_at_20_std
      value: -31.036826432373267
    - type: nauc_mrr_at_3_diff1
      value: 12.294460932554776
    - type: nauc_mrr_at_3_max
      value: -23.57847097970257
    - type: nauc_mrr_at_3_std
      value: -31.521356326067497
    - type: nauc_mrr_at_5_diff1
      value: 11.887561890187893
    - type: nauc_mrr_at_5_max
      value: -23.350381501385883
    - type: nauc_mrr_at_5_std
      value: -31.235803535943923
    - type: nauc_ndcg_at_1000_diff1
      value: 18.3264865443546
    - type: nauc_ndcg_at_1000_max
      value: -20.196278016245618
    - type: nauc_ndcg_at_1000_std
      value: -31.60715035240999
    - type: nauc_ndcg_at_100_diff1
      value: 18.347586002497316
    - type: nauc_ndcg_at_100_max
      value: -20.164411385553787
    - type: nauc_ndcg_at_100_std
      value: -31.688092183167104
    - type: nauc_ndcg_at_10_diff1
      value: 17.986012741495898
    - type: nauc_ndcg_at_10_max
      value: -19.194465099796407
    - type: nauc_ndcg_at_10_std
      value: -31.81139112072483
    - type: nauc_ndcg_at_1_diff1
      value: 20.17612594754981
    - type: nauc_ndcg_at_1_max
      value: -23.058659883369618
    - type: nauc_ndcg_at_1_std
      value: -31.028464230728563
    - type: nauc_ndcg_at_20_diff1
      value: 18.195224288342825
    - type: nauc_ndcg_at_20_max
      value: -19.93850679910986
    - type: nauc_ndcg_at_20_std
      value: -31.89304901101106
    - type: nauc_ndcg_at_3_diff1
      value: 17.98650612079755
    - type: nauc_ndcg_at_3_max
      value: -18.992887381655695
    - type: nauc_ndcg_at_3_std
      value: -32.02615903476361
    - type: nauc_ndcg_at_5_diff1
      value: 17.852873151866184
    - type: nauc_ndcg_at_5_max
      value: -18.37667538324993
    - type: nauc_ndcg_at_5_std
      value: -31.485058812560553
    - type: nauc_precision_at_1000_diff1
      value: 28.13719829218545
    - type: nauc_precision_at_1000_max
      value: 18.24025472404024
    - type: nauc_precision_at_1000_std
      value: 38.80401129796597
    - type: nauc_precision_at_100_diff1
      value: 32.68275932782318
    - type: nauc_precision_at_100_max
      value: 21.168881300935126
    - type: nauc_precision_at_100_std
      value: 3.3584681799290874
    - type: nauc_precision_at_10_diff1
      value: 11.093146862296395
    - type: nauc_precision_at_10_max
      value: 7.134120300431033
    - type: nauc_precision_at_10_std
      value: -31.277603693648775
    - type: nauc_precision_at_1_diff1
      value: 20.17612594754981
    - type: nauc_precision_at_1_max
      value: -23.058659883369618
    - type: nauc_precision_at_1_std
      value: -31.028464230728563
    - type: nauc_precision_at_20_diff1
      value: 10.959423455422453
    - type: nauc_precision_at_20_max
      value: 11.22070658323722
    - type: nauc_precision_at_20_std
      value: -34.837460859356554
    - type: nauc_precision_at_3_diff1
      value: 16.555833326617346
    - type: nauc_precision_at_3_max
      value: -12.995382532616235
    - type: nauc_precision_at_3_std
      value: -32.3954391909081
    - type: nauc_precision_at_5_diff1
      value: 15.056601641001969
    - type: nauc_precision_at_5_max
      value: -5.330719815786856
    - type: nauc_precision_at_5_std
      value: -29.322255770784206
    - type: nauc_recall_at_1000_diff1
      value: 28.137198292180138
    - type: nauc_recall_at_1000_max
      value: 18.240254724037225
    - type: nauc_recall_at_1000_std
      value: 38.80401129796458
    - type: nauc_recall_at_100_diff1
      value: 32.682759327819745
    - type: nauc_recall_at_100_max
      value: 21.168881300934643
    - type: nauc_recall_at_100_std
      value: 3.358468179927671
    - type: nauc_recall_at_10_diff1
      value: 11.093146862297438
    - type: nauc_recall_at_10_max
      value: 7.134120300431921
    - type: nauc_recall_at_10_std
      value: -31.277603693647716
    - type: nauc_recall_at_1_diff1
      value: 20.17612594754981
    - type: nauc_recall_at_1_max
      value: -23.058659883369618
    - type: nauc_recall_at_1_std
      value: -31.028464230728563
    - type: nauc_recall_at_20_diff1
      value: 10.959423455423444
    - type: nauc_recall_at_20_max
      value: 11.22070658323956
    - type: nauc_recall_at_20_std
      value: -34.83746085935601
    - type: nauc_recall_at_3_diff1
      value: 16.55583332661736
    - type: nauc_recall_at_3_max
      value: -12.995382532616167
    - type: nauc_recall_at_3_std
      value: -32.39543919090797
    - type: nauc_recall_at_5_diff1
      value: 15.056601641001993
    - type: nauc_recall_at_5_max
      value: -5.330719815786854
    - type: nauc_recall_at_5_std
      value: -29.322255770784167
    - type: ndcg_at_1
      value: 53.129000000000005
    - type: ndcg_at_10
      value: 75.285
    - type: ndcg_at_100
      value: 76.018
    - type: ndcg_at_1000
      value: 76.029
    - type: ndcg_at_20
      value: 75.869
    - type: ndcg_at_3
      value: 69.25099999999999
    - type: ndcg_at_5
      value: 72.922
    - type: precision_at_1
      value: 53.129000000000005
    - type: precision_at_10
      value: 9.658999999999999
    - type: precision_at_100
      value: 0.996
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_20
      value: 4.9399999999999995
    - type: precision_at_3
      value: 26.837
    - type: precision_at_5
      value: 17.866
    - type: recall_at_1
      value: 53.129000000000005
    - type: recall_at_10
      value: 96.586
    - type: recall_at_100
      value: 99.57300000000001
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_20
      value: 98.791
    - type: recall_at_3
      value: 80.512
    - type: recall_at_5
      value: 89.331
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ArxivClusteringP2P (default)
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
      split: test
      type: mteb/arxiv-clustering-p2p
    metrics:
    - type: main_score
      value: 53.16884944246256
    - type: v_measure
      value: 53.16884944246256
    - type: v_measure_std
      value: 14.33832089476221
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB ArxivClusteringS2S (default)
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
      split: test
      type: mteb/arxiv-clustering-s2s
    metrics:
    - type: main_score
      value: 46.86373219433518
    - type: v_measure
      value: 46.86373219433518
    - type: v_measure_std
      value: 14.720157485686848
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB AskUbuntuDupQuestions (default)
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
      split: test
      type: mteb/askubuntudupquestions-reranking
    metrics:
    - type: main_score
      value: 64.64337598763
    - type: map
      value: 64.64337598763
    - type: mrr
      value: 77.41535857186827
    - type: nAUC_map_diff1
      value: 19.39271950585529
    - type: nAUC_map_max
      value: 29.36178461695495
    - type: nAUC_map_std
      value: 15.19513250762444
    - type: nAUC_mrr_diff1
      value: 33.50457655525995
    - type: nAUC_mrr_max
      value: 43.15612390662652
    - type: nAUC_mrr_std
      value: 21.332057219123932
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB BIOSSES (default)
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
      split: test
      type: mteb/biosses-sts
    metrics:
    - type: cosine_pearson
      value: 88.29618951820042
    - type: cosine_spearman
      value: 86.8718589523104
    - type: euclidean_pearson
      value: 57.879315495228546
    - type: euclidean_spearman
      value: 59.70136033599529
    - type: main_score
      value: 86.8718589523104
    - type: manhattan_pearson
      value: 58.026039945855814
    - type: manhattan_spearman
      value: 59.548344801239764
    - type: pearson
      value: 88.29618951820042
    - type: spearman
      value: 86.8718589523104
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB Banking77Classification (default)
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
      split: test
      type: mteb/banking77
    metrics:
    - type: accuracy
      value: 90.16883116883116
    - type: f1
      value: 90.16172297880631
    - type: f1_weighted
      value: 90.16172297880631
    - type: main_score
      value: 90.16883116883116
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB BiorxivClusteringP2P (default)
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
      split: test
      type: mteb/biorxiv-clustering-p2p
    metrics:
    - type: main_score
      value: 49.37869478810584
    - type: v_measure
      value: 49.37869478810584
    - type: v_measure_std
      value: 0.8338525404588132
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB BiorxivClusteringS2S (default)
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
      split: test
      type: mteb/biorxiv-clustering-s2s
    metrics:
    - type: main_score
      value: 44.153287486944926
    - type: v_measure
      value: 44.153287486944926
    - type: v_measure_std
      value: 0.5615647069894507
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB CQADupstackAndroidRetrieval (default)
      revision: f46a197baaae43b4f621051089b82a364682dfeb
      split: test
      type: mteb/cqadupstack-android
    metrics:
    - type: main_score
      value: 52.315999999999995
    - type: map_at_1
      value: 33.641
    - type: map_at_10
      value: 45.689
    - type: map_at_100
      value: 47.313
    - type: map_at_1000
      value: 47.417
    - type: map_at_20
      value: 46.608
    - type: map_at_3
      value: 42.175000000000004
    - type: map_at_5
      value: 43.891000000000005
    - type: mrr_at_1
      value: 40.772532188841204
    - type: mrr_at_10
      value: 51.509185457683316
    - type: mrr_at_100
      value: 52.220484021166236
    - type: mrr_at_1000
      value: 52.252809733745146
    - type: mrr_at_20
      value: 51.92754479836423
    - type: mrr_at_3
      value: 49.1654744873629
    - type: mrr_at_5
      value: 50.20982355746304
    - type: nauc_map_at_1000_diff1
      value: 50.031087935331655
    - type: nauc_map_at_1000_max
      value: 43.53637528147624
    - type: nauc_map_at_1000_std
      value: -5.689030253852926
    - type: nauc_map_at_100_diff1
      value: 50.04662900414012
    - type: nauc_map_at_100_max
      value: 43.53508350684863
    - type: nauc_map_at_100_std
      value: -5.674327991538789
    - type: nauc_map_at_10_diff1
      value: 50.44461151089684
    - type: nauc_map_at_10_max
      value: 43.49810943631597
    - type: nauc_map_at_10_std
      value: -6.627723012937466
    - type: nauc_map_at_1_diff1
      value: 55.300717032500756
    - type: nauc_map_at_1_max
      value: 40.712611817162106
    - type: nauc_map_at_1_std
      value: -7.382978845041299
    - type: nauc_map_at_20_diff1
      value: 50.1174442568897
    - type: nauc_map_at_20_max
      value: 43.5811062476766
    - type: nauc_map_at_20_std
      value: -5.7759730811512915
    - type: nauc_map_at_3_diff1
      value: 52.00729479691187
    - type: nauc_map_at_3_max
      value: 42.4347799369893
    - type: nauc_map_at_3_std
      value: -7.597568167730342
    - type: nauc_map_at_5_diff1
      value: 50.79895759161379
    - type: nauc_map_at_5_max
      value: 43.4553521540224
    - type: nauc_map_at_5_std
      value: -7.005488103650704
    - type: nauc_mrr_at_1000_diff1
      value: 48.369412664775375
    - type: nauc_mrr_at_1000_max
      value: 43.447512142202186
    - type: nauc_mrr_at_1000_std
      value: -3.7554010485202327
    - type: nauc_mrr_at_100_diff1
      value: 48.367155650696596
    - type: nauc_mrr_at_100_max
      value: 43.43672113468781
    - type: nauc_mrr_at_100_std
      value: -3.7584259547352685
    - type: nauc_mrr_at_10_diff1
      value: 48.22399231961935
    - type: nauc_mrr_at_10_max
      value: 43.446120963764464
    - type: nauc_mrr_at_10_std
      value: -3.864378526696108
    - type: nauc_mrr_at_1_diff1
      value: 51.48564590019635
    - type: nauc_mrr_at_1_max
      value: 44.62171085012241
    - type: nauc_mrr_at_1_std
      value: -3.818072138180037
    - type: nauc_mrr_at_20_diff1
      value: 48.302896769214406
    - type: nauc_mrr_at_20_max
      value: 43.44469481144226
    - type: nauc_mrr_at_20_std
      value: -3.6988037745270796
    - type: nauc_mrr_at_3_diff1
      value: 49.24730476322711
    - type: nauc_mrr_at_3_max
      value: 43.46926120369678
    - type: nauc_mrr_at_3_std
      value: -4.40710875057783
    - type: nauc_mrr_at_5_diff1
      value: 48.312344317357486
    - type: nauc_mrr_at_5_max
      value: 43.852651171669024
    - type: nauc_mrr_at_5_std
      value: -3.866500929216681
    - type: nauc_ndcg_at_1000_diff1
      value: 48.20219427390479
    - type: nauc_ndcg_at_1000_max
      value: 43.65089095895309
    - type: nauc_ndcg_at_1000_std
      value: -3.3158697105104538
    - type: nauc_ndcg_at_100_diff1
      value: 48.21515564601719
    - type: nauc_ndcg_at_100_max
      value: 43.79417151116643
    - type: nauc_ndcg_at_100_std
      value: -2.9711870696304046
    - type: nauc_ndcg_at_10_diff1
      value: 48.07133753085607
    - type: nauc_ndcg_at_10_max
      value: 43.49490777323681
    - type: nauc_ndcg_at_10_std
      value: -5.3066358267446425
    - type: nauc_ndcg_at_1_diff1
      value: 51.48564590019635
    - type: nauc_ndcg_at_1_max
      value: 44.62171085012241
    - type: nauc_ndcg_at_1_std
      value: -3.818072138180037
    - type: nauc_ndcg_at_20_diff1
      value: 47.63812787566806
    - type: nauc_ndcg_at_20_max
      value: 43.53447921487485
    - type: nauc_ndcg_at_20_std
      value: -3.6391468425108466
    - type: nauc_ndcg_at_3_diff1
      value: 50.05079774447682
    - type: nauc_ndcg_at_3_max
      value: 42.55676504485692
    - type: nauc_ndcg_at_3_std
      value: -6.399487773494622
    - type: nauc_ndcg_at_5_diff1
      value: 48.45632392122646
    - type: nauc_ndcg_at_5_max
      value: 43.62893544032143
    - type: nauc_ndcg_at_5_std
      value: -5.554511079346528
    - type: nauc_precision_at_1000_diff1
      value: -25.27127937599737
    - type: nauc_precision_at_1000_max
      value: -14.93813752145489
    - type: nauc_precision_at_1000_std
      value: -0.3787281907430783
    - type: nauc_precision_at_100_diff1
      value: -17.80925394382853
    - type: nauc_precision_at_100_max
      value: -3.6761849827727406
    - type: nauc_precision_at_100_std
      value: 7.841673059322428
    - type: nauc_precision_at_10_diff1
      value: 2.726200066807412
    - type: nauc_precision_at_10_max
      value: 17.962136609175136
    - type: nauc_precision_at_10_std
      value: 5.729763148982194
    - type: nauc_precision_at_1_diff1
      value: 51.48564590019635
    - type: nauc_precision_at_1_max
      value: 44.62171085012241
    - type: nauc_precision_at_1_std
      value: -3.818072138180037
    - type: nauc_precision_at_20_diff1
      value: -6.487164292001067
    - type: nauc_precision_at_20_max
      value: 9.778356281586516
    - type: nauc_precision_at_20_std
      value: 9.604010389554679
    - type: nauc_precision_at_3_diff1
      value: 28.858897720773374
    - type: nauc_precision_at_3_max
      value: 34.91220154225439
    - type: nauc_precision_at_3_std
      value: -1.6150283438909432
    - type: nauc_precision_at_5_diff1
      value: 16.10580579242152
    - type: nauc_precision_at_5_max
      value: 29.64713521720959
    - type: nauc_precision_at_5_std
      value: 2.4132129711626327
    - type: nauc_recall_at_1000_diff1
      value: 35.488751910757976
    - type: nauc_recall_at_1000_max
      value: 56.648265097548524
    - type: nauc_recall_at_1000_std
      value: 49.399000259977186
    - type: nauc_recall_at_100_diff1
      value: 40.850907747191144
    - type: nauc_recall_at_100_max
      value: 45.77637959778657
    - type: nauc_recall_at_100_std
      value: 17.387665528465522
    - type: nauc_recall_at_10_diff1
      value: 40.30847417344133
    - type: nauc_recall_at_10_max
      value: 40.05523748427962
    - type: nauc_recall_at_10_std
      value: -4.215435715888072
    - type: nauc_recall_at_1_diff1
      value: 55.300717032500756
    - type: nauc_recall_at_1_max
      value: 40.712611817162106
    - type: nauc_recall_at_1_std
      value: -7.382978845041299
    - type: nauc_recall_at_20_diff1
      value: 37.21696912115577
    - type: nauc_recall_at_20_max
      value: 39.92721076833355
    - type: nauc_recall_at_20_std
      value: 2.4034636628070105
    - type: nauc_recall_at_3_diff1
      value: 48.417789387438006
    - type: nauc_recall_at_3_max
      value: 39.90602671619894
    - type: nauc_recall_at_3_std
      value: -7.155995719055378
    - type: nauc_recall_at_5_diff1
      value: 43.661772766088916
    - type: nauc_recall_at_5_max
      value: 41.51830600574735
    - type: nauc_recall_at_5_std
      value: -5.060725402497923
    - type: ndcg_at_1
      value: 40.772999999999996
    - type: ndcg_at_10
      value: 52.315999999999995
    - type: ndcg_at_100
      value: 57.772999999999996
    - type: ndcg_at_1000
      value: 59.258
    - type: ndcg_at_20
      value: 54.468
    - type: ndcg_at_3
      value: 47.471999999999994
    - type: ndcg_at_5
      value: 49.132999999999996
    - type: precision_at_1
      value: 40.772999999999996
    - type: precision_at_10
      value: 10.043000000000001
    - type: precision_at_100
      value: 1.6019999999999999
    - type: precision_at_1000
      value: 0.2
    - type: precision_at_20
      value: 5.987
    - type: precision_at_3
      value: 22.938
    - type: precision_at_5
      value: 16.08
    - type: recall_at_1
      value: 33.641
    - type: recall_at_10
      value: 65.083
    - type: recall_at_100
      value: 87.556
    - type: recall_at_1000
      value: 96.654
    - type: recall_at_20
      value: 72.557
    - type: recall_at_3
      value: 50.455000000000005
    - type: recall_at_5
      value: 55.588
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackEnglishRetrieval (default)
      revision: ad9991cb51e31e31e430383c75ffb2885547b5f0
      split: test
      type: mteb/cqadupstack-english
    metrics:
    - type: main_score
      value: 51.343
    - type: map_at_1
      value: 34.593
    - type: map_at_10
      value: 45.547
    - type: map_at_100
      value: 46.816
    - type: map_at_1000
      value: 46.943
    - type: map_at_20
      value: 46.253
    - type: map_at_3
      value: 42.583
    - type: map_at_5
      value: 44.304
    - type: mrr_at_1
      value: 43.248407643312106
    - type: mrr_at_10
      value: 51.7940804772015
    - type: mrr_at_100
      value: 52.39883932510644
    - type: mrr_at_1000
      value: 52.43879811294841
    - type: mrr_at_20
      value: 52.172418353372684
    - type: mrr_at_3
      value: 49.925690021231425
    - type: mrr_at_5
      value: 51.09447983014862
    - type: nauc_map_at_1000_diff1
      value: 48.67335443556064
    - type: nauc_map_at_1000_max
      value: 37.822985028240154
    - type: nauc_map_at_1000_std
      value: -8.900573979747413
    - type: nauc_map_at_100_diff1
      value: 48.698202225769464
    - type: nauc_map_at_100_max
      value: 37.738023301565356
    - type: nauc_map_at_100_std
      value: -9.043636874625522
    - type: nauc_map_at_10_diff1
      value: 48.71921251874774
    - type: nauc_map_at_10_max
      value: 36.696516553601356
    - type: nauc_map_at_10_std
      value: -10.66232883806279
    - type: nauc_map_at_1_diff1
      value: 54.77522470457049
    - type: nauc_map_at_1_max
      value: 30.840291333860343
    - type: nauc_map_at_1_std
      value: -13.666953809326712
    - type: nauc_map_at_20_diff1
      value: 48.76147318325276
    - type: nauc_map_at_20_max
      value: 37.156719457662916
    - type: nauc_map_at_20_std
      value: -9.86719626130413
    - type: nauc_map_at_3_diff1
      value: 50.18867450904811
    - type: nauc_map_at_3_max
      value: 35.37817456543256
    - type: nauc_map_at_3_std
      value: -12.93988735522222
    - type: nauc_map_at_5_diff1
      value: 48.99361218309583
    - type: nauc_map_at_5_max
      value: 35.90269875806009
    - type: nauc_map_at_5_std
      value: -12.004877184865785
    - type: nauc_mrr_at_1000_diff1
      value: 47.77079679787741
    - type: nauc_mrr_at_1000_max
      value: 41.33549585329522
    - type: nauc_mrr_at_1000_std
      value: -3.318672526140045
    - type: nauc_mrr_at_100_diff1
      value: 47.77323135207226
    - type: nauc_mrr_at_100_max
      value: 41.35043932569742
    - type: nauc_mrr_at_100_std
      value: -3.31292522098766
    - type: nauc_mrr_at_10_diff1
      value: 47.639375553044175
    - type: nauc_mrr_at_10_max
      value: 41.295865992298005
    - type: nauc_mrr_at_10_std
      value: -3.548901132097741
    - type: nauc_mrr_at_1_diff1
      value: 53.21585888474443
    - type: nauc_mrr_at_1_max
      value: 41.264820758056736
    - type: nauc_mrr_at_1_std
      value: -4.157978647993681
    - type: nauc_mrr_at_20_diff1
      value: 47.73992536914945
    - type: nauc_mrr_at_20_max
      value: 41.25857342302041
    - type: nauc_mrr_at_20_std
      value: -3.4353806455518194
    - type: nauc_mrr_at_3_diff1
      value: 47.986921399252374
    - type: nauc_mrr_at_3_max
      value: 41.22599517238877
    - type: nauc_mrr_at_3_std
      value: -4.170152543664638
    - type: nauc_mrr_at_5_diff1
      value: 47.68203368915962
    - type: nauc_mrr_at_5_max
      value: 41.20159856834227
    - type: nauc_mrr_at_5_std
      value: -3.8516603464270918
    - type: nauc_ndcg_at_1000_diff1
      value: 46.30081465952772
    - type: nauc_ndcg_at_1000_max
      value: 39.864403084833185
    - type: nauc_ndcg_at_1000_std
      value: -3.5886667363187477
    - type: nauc_ndcg_at_100_diff1
      value: 46.37475981566266
    - type: nauc_ndcg_at_100_max
      value: 39.882135392553344
    - type: nauc_ndcg_at_100_std
      value: -4.046056305972922
    - type: nauc_ndcg_at_10_diff1
      value: 46.1108552726571
    - type: nauc_ndcg_at_10_max
      value: 38.89924272656985
    - type: nauc_ndcg_at_10_std
      value: -6.988162257454499
    - type: nauc_ndcg_at_1_diff1
      value: 53.21585888474443
    - type: nauc_ndcg_at_1_max
      value: 41.264820758056736
    - type: nauc_ndcg_at_1_std
      value: -4.157978647993681
    - type: nauc_ndcg_at_20_diff1
      value: 46.53689482819702
    - type: nauc_ndcg_at_20_max
      value: 38.98429546252361
    - type: nauc_ndcg_at_20_std
      value: -6.17295795707756
    - type: nauc_ndcg_at_3_diff1
      value: 46.87441388826659
    - type: nauc_ndcg_at_3_max
      value: 39.509987074429226
    - type: nauc_ndcg_at_3_std
      value: -7.211087569154668
    - type: nauc_ndcg_at_5_diff1
      value: 46.21276772949452
    - type: nauc_ndcg_at_5_max
      value: 38.693404358670335
    - type: nauc_ndcg_at_5_std
      value: -7.935343281737274
    - type: nauc_precision_at_1000_diff1
      value: -15.035625997115945
    - type: nauc_precision_at_1000_max
      value: 17.609429074537303
    - type: nauc_precision_at_1000_std
      value: 33.50216908946751
    - type: nauc_precision_at_100_diff1
      value: -8.577825109805744
    - type: nauc_precision_at_100_max
      value: 27.908142881704194
    - type: nauc_precision_at_100_std
      value: 37.37502674795575
    - type: nauc_precision_at_10_diff1
      value: 4.66621879837823
    - type: nauc_precision_at_10_max
      value: 35.296979975828854
    - type: nauc_precision_at_10_std
      value: 21.929955057403305
    - type: nauc_precision_at_1_diff1
      value: 53.21585888474443
    - type: nauc_precision_at_1_max
      value: 41.264820758056736
    - type: nauc_precision_at_1_std
      value: -4.157978647993681
    - type: nauc_precision_at_20_diff1
      value: 0.282744932044692
    - type: nauc_precision_at_20_max
      value: 32.33807502825822
    - type: nauc_precision_at_20_std
      value: 28.256817127792967
    - type: nauc_precision_at_3_diff1
      value: 22.99372404315396
    - type: nauc_precision_at_3_max
      value: 40.77204813605887
    - type: nauc_precision_at_3_std
      value: 7.794641697217284
    - type: nauc_precision_at_5_diff1
      value: 13.11635711185126
    - type: nauc_precision_at_5_max
      value: 37.578046404266395
    - type: nauc_precision_at_5_std
      value: 13.974528031618814
    - type: nauc_recall_at_1000_diff1
      value: 30.70502401820734
    - type: nauc_recall_at_1000_max
      value: 39.45352106327067
    - type: nauc_recall_at_1000_std
      value: 27.199209908129625
    - type: nauc_recall_at_100_diff1
      value: 35.476479923805535
    - type: nauc_recall_at_100_max
      value: 40.07230782745946
    - type: nauc_recall_at_100_std
      value: 11.017688373969243
    - type: nauc_recall_at_10_diff1
      value: 37.959597598149976
    - type: nauc_recall_at_10_max
      value: 34.805900488852636
    - type: nauc_recall_at_10_std
      value: -7.513194356128469
    - type: nauc_recall_at_1_diff1
      value: 54.77522470457049
    - type: nauc_recall_at_1_max
      value: 30.840291333860343
    - type: nauc_recall_at_1_std
      value: -13.666953809326712
    - type: nauc_recall_at_20_diff1
      value: 38.04488508789386
    - type: nauc_recall_at_20_max
      value: 35.31449482267666
    - type: nauc_recall_at_20_std
      value: -3.573723664596999
    - type: nauc_recall_at_3_diff1
      value: 43.079600909307466
    - type: nauc_recall_at_3_max
      value: 33.722222410363685
    - type: nauc_recall_at_3_std
      value: -13.139450781078693
    - type: nauc_recall_at_5_diff1
      value: 39.75224622281559
    - type: nauc_recall_at_5_max
      value: 33.32195498979599
    - type: nauc_recall_at_5_std
      value: -11.939404508299996
    - type: ndcg_at_1
      value: 43.248
    - type: ndcg_at_10
      value: 51.343
    - type: ndcg_at_100
      value: 55.421
    - type: ndcg_at_1000
      value: 57.282999999999994
    - type: ndcg_at_20
      value: 52.954
    - type: ndcg_at_3
      value: 47.766
    - type: ndcg_at_5
      value: 49.498999999999995
    - type: precision_at_1
      value: 43.248
    - type: precision_at_10
      value: 9.522
    - type: precision_at_100
      value: 1.494
    - type: precision_at_1000
      value: 0.194
    - type: precision_at_20
      value: 5.545
    - type: precision_at_3
      value: 23.163
    - type: precision_at_5
      value: 16.14
    - type: recall_at_1
      value: 34.593
    - type: recall_at_10
      value: 60.870999999999995
    - type: recall_at_100
      value: 78.118
    - type: recall_at_1000
      value: 89.693
    - type: recall_at_20
      value: 66.866
    - type: recall_at_3
      value: 49.51
    - type: recall_at_5
      value: 54.82
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackGamingRetrieval (default)
      revision: 4885aa143210c98657558c04aaf3dc47cfb54340
      split: test
      type: mteb/cqadupstack-gaming
    metrics:
    - type: main_score
      value: 56.730000000000004
    - type: map_at_1
      value: 38.879000000000005
    - type: map_at_10
      value: 50.902
    - type: map_at_100
      value: 51.913
    - type: map_at_1000
      value: 51.974
    - type: map_at_20
      value: 51.485
    - type: map_at_3
      value: 47.542
    - type: map_at_5
      value: 49.555
    - type: mrr_at_1
      value: 44.5141065830721
    - type: mrr_at_10
      value: 54.33761257899189
    - type: mrr_at_100
      value: 54.976838027222705
    - type: mrr_at_1000
      value: 55.01349256343839
    - type: mrr_at_20
      value: 54.70610741776682
    - type: mrr_at_3
      value: 51.78683385579937
    - type: mrr_at_5
      value: 53.37931034482758
    - type: nauc_map_at_1000_diff1
      value: 50.64299899142634
    - type: nauc_map_at_1000_max
      value: 36.698035080321326
    - type: nauc_map_at_1000_std
      value: -3.756494285833674
    - type: nauc_map_at_100_diff1
      value: 50.622793159211774
    - type: nauc_map_at_100_max
      value: 36.692850786079575
    - type: nauc_map_at_100_std
      value: -3.7485394000958703
    - type: nauc_map_at_10_diff1
      value: 50.55187180590186
    - type: nauc_map_at_10_max
      value: 36.10664794042924
    - type: nauc_map_at_10_std
      value: -4.632826967133337
    - type: nauc_map_at_1_diff1
      value: 55.05750881444125
    - type: nauc_map_at_1_max
      value: 32.60063181438657
    - type: nauc_map_at_1_std
      value: -7.724648821775435
    - type: nauc_map_at_20_diff1
      value: 50.63016534099507
    - type: nauc_map_at_20_max
      value: 36.494850275793276
    - type: nauc_map_at_20_std
      value: -4.061837340919665
    - type: nauc_map_at_3_diff1
      value: 50.99634496573267
    - type: nauc_map_at_3_max
      value: 34.683061650442404
    - type: nauc_map_at_3_std
      value: -6.618466533361132
    - type: nauc_map_at_5_diff1
      value: 50.55471319386312
    - type: nauc_map_at_5_max
      value: 35.73463998730437
    - type: nauc_map_at_5_std
      value: -5.31799289256796
    - type: nauc_mrr_at_1000_diff1
      value: 50.68660123518591
    - type: nauc_mrr_at_1000_max
      value: 37.721680351330036
    - type: nauc_mrr_at_1000_std
      value: -3.021512924039592
    - type: nauc_mrr_at_100_diff1
      value: 50.67549688978347
    - type: nauc_mrr_at_100_max
      value: 37.73588231555737
    - type: nauc_mrr_at_100_std
      value: -2.9882989335927346
    - type: nauc_mrr_at_10_diff1
      value: 50.51528784437163
    - type: nauc_mrr_at_10_max
      value: 37.47356367375625
    - type: nauc_mrr_at_10_std
      value: -3.339957801672487
    - type: nauc_mrr_at_1_diff1
      value: 55.15096230434279
    - type: nauc_mrr_at_1_max
      value: 36.72571386341383
    - type: nauc_mrr_at_1_std
      value: -5.9794263311551745
    - type: nauc_mrr_at_20_diff1
      value: 50.610714355036656
    - type: nauc_mrr_at_20_max
      value: 37.70305389603564
    - type: nauc_mrr_at_20_std
      value: -3.072300181201535
    - type: nauc_mrr_at_3_diff1
      value: 50.76218695783636
    - type: nauc_mrr_at_3_max
      value: 37.12200218567894
    - type: nauc_mrr_at_3_std
      value: -4.375808935158392
    - type: nauc_mrr_at_5_diff1
      value: 50.5439022293488
    - type: nauc_mrr_at_5_max
      value: 37.552059182545996
    - type: nauc_mrr_at_5_std
      value: -3.6109501843729506
    - type: nauc_ndcg_at_1000_diff1
      value: 49.76605435140743
    - type: nauc_ndcg_at_1000_max
      value: 38.45072714945938
    - type: nauc_ndcg_at_1000_std
      value: -0.7556311767251159
    - type: nauc_ndcg_at_100_diff1
      value: 49.35858064236287
    - type: nauc_ndcg_at_100_max
      value: 38.82348019704706
    - type: nauc_ndcg_at_100_std
      value: 0.1267226726383699
    - type: nauc_ndcg_at_10_diff1
      value: 48.71303182912485
    - type: nauc_ndcg_at_10_max
      value: 37.183467158050334
    - type: nauc_ndcg_at_10_std
      value: -2.2690079683429514
    - type: nauc_ndcg_at_1_diff1
      value: 55.15096230434279
    - type: nauc_ndcg_at_1_max
      value: 36.72571386341383
    - type: nauc_ndcg_at_1_std
      value: -5.9794263311551745
    - type: nauc_ndcg_at_20_diff1
      value: 48.96937586821276
    - type: nauc_ndcg_at_20_max
      value: 38.00686643814116
    - type: nauc_ndcg_at_20_std
      value: -1.0205120300633659
    - type: nauc_ndcg_at_3_diff1
      value: 49.54317719689958
    - type: nauc_ndcg_at_3_max
      value: 35.683711597485626
    - type: nauc_ndcg_at_3_std
      value: -5.215106681638484
    - type: nauc_ndcg_at_5_diff1
      value: 48.86989255377164
    - type: nauc_ndcg_at_5_max
      value: 36.99055154118703
    - type: nauc_ndcg_at_5_std
      value: -3.3744491989320573
    - type: nauc_precision_at_1000_diff1
      value: -9.3980232166607
    - type: nauc_precision_at_1000_max
      value: 14.691976949282054
    - type: nauc_precision_at_1000_std
      value: 23.233941524964404
    - type: nauc_precision_at_100_diff1
      value: -3.948701209864356
    - type: nauc_precision_at_100_max
      value: 22.256352399415093
    - type: nauc_precision_at_100_std
      value: 26.901091669184833
    - type: nauc_precision_at_10_diff1
      value: 14.314341369117708
    - type: nauc_precision_at_10_max
      value: 29.11276969572707
    - type: nauc_precision_at_10_std
      value: 13.795063294330784
    - type: nauc_precision_at_1_diff1
      value: 55.15096230434279
    - type: nauc_precision_at_1_max
      value: 36.72571386341383
    - type: nauc_precision_at_1_std
      value: -5.9794263311551745
    - type: nauc_precision_at_20_diff1
      value: 7.982841936037328
    - type: nauc_precision_at_20_max
      value: 27.68248112221504
    - type: nauc_precision_at_20_std
      value: 19.859120817525287
    - type: nauc_precision_at_3_diff1
      value: 31.57629768677489
    - type: nauc_precision_at_3_max
      value: 33.9050359424042
    - type: nauc_precision_at_3_std
      value: 1.2394592558954534
    - type: nauc_precision_at_5_diff1
      value: 22.581242632344072
    - type: nauc_precision_at_5_max
      value: 33.125378486548044
    - type: nauc_precision_at_5_std
      value: 8.130818770875205
    - type: nauc_recall_at_1000_diff1
      value: 45.10175276662796
    - type: nauc_recall_at_1000_max
      value: 55.78601062989057
    - type: nauc_recall_at_1000_std
      value: 50.26989332506703
    - type: nauc_recall_at_100_diff1
      value: 40.43279094297972
    - type: nauc_recall_at_100_max
      value: 50.32502226244754
    - type: nauc_recall_at_100_std
      value: 30.50067477968058
    - type: nauc_recall_at_10_diff1
      value: 40.273769431383435
    - type: nauc_recall_at_10_max
      value: 36.49724884555957
    - type: nauc_recall_at_10_std
      value: 3.2218721583264895
    - type: nauc_recall_at_1_diff1
      value: 55.05750881444125
    - type: nauc_recall_at_1_max
      value: 32.60063181438657
    - type: nauc_recall_at_1_std
      value: -7.724648821775435
    - type: nauc_recall_at_20_diff1
      value: 40.36167281217596
    - type: nauc_recall_at_20_max
      value: 40.41420275995583
    - type: nauc_recall_at_20_std
      value: 10.597913347328813
    - type: nauc_recall_at_3_diff1
      value: 44.75300028813483
    - type: nauc_recall_at_3_max
      value: 33.88475963435326
    - type: nauc_recall_at_3_std
      value: -5.4508036619593545
    - type: nauc_recall_at_5_diff1
      value: 42.04884424961388
    - type: nauc_recall_at_5_max
      value: 36.17614101715584
    - type: nauc_recall_at_5_std
      value: -0.8874899973464042
    - type: ndcg_at_1
      value: 44.513999999999996
    - type: ndcg_at_10
      value: 56.730000000000004
    - type: ndcg_at_100
      value: 60.809999999999995
    - type: ndcg_at_1000
      value: 62.075
    - type: ndcg_at_20
      value: 58.36
    - type: ndcg_at_3
      value: 51.12
    - type: ndcg_at_5
      value: 54.112
    - type: precision_at_1
      value: 44.513999999999996
    - type: precision_at_10
      value: 9.103
    - type: precision_at_100
      value: 1.208
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_20
      value: 5.06
    - type: precision_at_3
      value: 22.633
    - type: precision_at_5
      value: 15.712000000000002
    - type: recall_at_1
      value: 38.879000000000005
    - type: recall_at_10
      value: 70.473
    - type: recall_at_100
      value: 88.319
    - type: recall_at_1000
      value: 97.184
    - type: recall_at_20
      value: 76.411
    - type: recall_at_3
      value: 55.745
    - type: recall_at_5
      value: 63.071999999999996
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackGisRetrieval (default)
      revision: 5003b3064772da1887988e05400cf3806fe491f2
      split: test
      type: mteb/cqadupstack-gis
    metrics:
    - type: main_score
      value: 38.605000000000004
    - type: map_at_1
      value: 25.635
    - type: map_at_10
      value: 33.852
    - type: map_at_100
      value: 34.964
    - type: map_at_1000
      value: 35.045
    - type: map_at_20
      value: 34.483000000000004
    - type: map_at_3
      value: 31.455
    - type: map_at_5
      value: 32.857
    - type: mrr_at_1
      value: 27.570621468926554
    - type: mrr_at_10
      value: 36.023988879921085
    - type: mrr_at_100
      value: 36.96909821562831
    - type: mrr_at_1000
      value: 37.02758399106491
    - type: mrr_at_20
      value: 36.54327764554978
    - type: mrr_at_3
      value: 33.69114877589454
    - type: mrr_at_5
      value: 35.07532956685499
    - type: nauc_map_at_1000_diff1
      value: 39.211781799139054
    - type: nauc_map_at_1000_max
      value: 32.55345475290884
    - type: nauc_map_at_1000_std
      value: -1.265215084741391
    - type: nauc_map_at_100_diff1
      value: 39.19052003812807
    - type: nauc_map_at_100_max
      value: 32.52932274042929
    - type: nauc_map_at_100_std
      value: -1.2890452039736608
    - type: nauc_map_at_10_diff1
      value: 39.04320229077609
    - type: nauc_map_at_10_max
      value: 32.58324790339162
    - type: nauc_map_at_10_std
      value: -1.5871496704850256
    - type: nauc_map_at_1_diff1
      value: 46.56298943416125
    - type: nauc_map_at_1_max
      value: 29.588884632253475
    - type: nauc_map_at_1_std
      value: -3.083164597327146
    - type: nauc_map_at_20_diff1
      value: 39.12946471211154
    - type: nauc_map_at_20_max
      value: 32.52951057102236
    - type: nauc_map_at_20_std
      value: -1.4322489477929932
    - type: nauc_map_at_3_diff1
      value: 39.84663705685742
    - type: nauc_map_at_3_max
      value: 33.49072622062519
    - type: nauc_map_at_3_std
      value: -1.1479245598447843
    - type: nauc_map_at_5_diff1
      value: 39.3936434273089
    - type: nauc_map_at_5_max
      value: 32.75018291103631
    - type: nauc_map_at_5_std
      value: -1.9299111782404523
    - type: nauc_mrr_at_1000_diff1
      value: 37.814221406708484
    - type: nauc_mrr_at_1000_max
      value: 33.24270423500106
    - type: nauc_mrr_at_1000_std
      value: -0.4462285423407193
    - type: nauc_mrr_at_100_diff1
      value: 37.77979910897116
    - type: nauc_mrr_at_100_max
      value: 33.238564560412016
    - type: nauc_mrr_at_100_std
      value: -0.45827426583512604
    - type: nauc_mrr_at_10_diff1
      value: 37.695288177194044
    - type: nauc_mrr_at_10_max
      value: 33.31456943076525
    - type: nauc_mrr_at_10_std
      value: -0.6986817832243588
    - type: nauc_mrr_at_1_diff1
      value: 45.00955245662916
    - type: nauc_mrr_at_1_max
      value: 31.26729884132408
    - type: nauc_mrr_at_1_std
      value: -1.9847042174178882
    - type: nauc_mrr_at_20_diff1
      value: 37.67346067235003
    - type: nauc_mrr_at_20_max
      value: 33.232158099696285
    - type: nauc_mrr_at_20_std
      value: -0.5051050412938318
    - type: nauc_mrr_at_3_diff1
      value: 38.36691271458178
    - type: nauc_mrr_at_3_max
      value: 34.34058210996929
    - type: nauc_mrr_at_3_std
      value: 0.2173867046243404
    - type: nauc_mrr_at_5_diff1
      value: 37.973722327328815
    - type: nauc_mrr_at_5_max
      value: 33.622494346385764
    - type: nauc_mrr_at_5_std
      value: -0.8988212795536653
    - type: nauc_ndcg_at_1000_diff1
      value: 36.59844568081433
    - type: nauc_ndcg_at_1000_max
      value: 32.875450773811195
    - type: nauc_ndcg_at_1000_std
      value: 0.9571717377866221
    - type: nauc_ndcg_at_100_diff1
      value: 36.00198688351201
    - type: nauc_ndcg_at_100_max
      value: 32.21453290589628
    - type: nauc_ndcg_at_100_std
      value: 0.6593320800081022
    - type: nauc_ndcg_at_10_diff1
      value: 35.96874481832434
    - type: nauc_ndcg_at_10_max
      value: 32.495670150366216
    - type: nauc_ndcg_at_10_std
      value: -0.9324209794892333
    - type: nauc_ndcg_at_1_diff1
      value: 45.00955245662916
    - type: nauc_ndcg_at_1_max
      value: 31.26729884132408
    - type: nauc_ndcg_at_1_std
      value: -1.9847042174178882
    - type: nauc_ndcg_at_20_diff1
      value: 36.03533734210292
    - type: nauc_ndcg_at_20_max
      value: 32.24375568627277
    - type: nauc_ndcg_at_20_std
      value: -0.3596391132919354
    - type: nauc_ndcg_at_3_diff1
      value: 37.56515351352606
    - type: nauc_ndcg_at_3_max
      value: 34.40909560361463
    - type: nauc_ndcg_at_3_std
      value: -0.13627845111454864
    - type: nauc_ndcg_at_5_diff1
      value: 36.600845693080785
    - type: nauc_ndcg_at_5_max
      value: 32.985955018025805
    - type: nauc_ndcg_at_5_std
      value: -1.6916565029717592
    - type: nauc_precision_at_1000_diff1
      value: -3.445116813097008
    - type: nauc_precision_at_1000_max
      value: 14.429823053173335
    - type: nauc_precision_at_1000_std
      value: 11.724477717097624
    - type: nauc_precision_at_100_diff1
      value: 8.397559719859894
    - type: nauc_precision_at_100_max
      value: 20.066139779586265
    - type: nauc_precision_at_100_std
      value: 10.580324704387824
    - type: nauc_precision_at_10_diff1
      value: 22.321578540672835
    - type: nauc_precision_at_10_max
      value: 30.583827524369816
    - type: nauc_precision_at_10_std
      value: 1.8521481605183971
    - type: nauc_precision_at_1_diff1
      value: 45.00955245662916
    - type: nauc_precision_at_1_max
      value: 31.26729884132408
    - type: nauc_precision_at_1_std
      value: -1.9847042174178882
    - type: nauc_precision_at_20_diff1
      value: 19.872934967880912
    - type: nauc_precision_at_20_max
      value: 27.126514701768027
    - type: nauc_precision_at_20_std
      value: 4.247074218167322
    - type: nauc_precision_at_3_diff1
      value: 29.427641663443254
    - type: nauc_precision_at_3_max
      value: 37.62617222609326
    - type: nauc_precision_at_3_std
      value: 3.948585303384073
    - type: nauc_precision_at_5_diff1
      value: 25.773033061011386
    - type: nauc_precision_at_5_max
      value: 34.06383588336501
    - type: nauc_precision_at_5_std
      value: -0.14029168831194636
    - type: nauc_recall_at_1000_diff1
      value: 17.639486925117307
    - type: nauc_recall_at_1000_max
      value: 36.34212631145863
    - type: nauc_recall_at_1000_std
      value: 31.800274452457263
    - type: nauc_recall_at_100_diff1
      value: 19.873163489274337
    - type: nauc_recall_at_100_max
      value: 25.80361327038133
    - type: nauc_recall_at_100_std
      value: 11.58255008771282
    - type: nauc_recall_at_10_diff1
      value: 27.060109398377406
    - type: nauc_recall_at_10_max
      value: 29.91693722293117
    - type: nauc_recall_at_10_std
      value: 0.18719788163565917
    - type: nauc_recall_at_1_diff1
      value: 46.56298943416125
    - type: nauc_recall_at_1_max
      value: 29.588884632253475
    - type: nauc_recall_at_1_std
      value: -3.083164597327146
    - type: nauc_recall_at_20_diff1
      value: 26.291713722640743
    - type: nauc_recall_at_20_max
      value: 28.515618290535024
    - type: nauc_recall_at_20_std
      value: 2.492294019599597
    - type: nauc_recall_at_3_diff1
      value: 31.489494073712326
    - type: nauc_recall_at_3_max
      value: 35.223490974009216
    - type: nauc_recall_at_3_std
      value: 1.2555042620982575
    - type: nauc_recall_at_5_diff1
      value: 29.198216149895355
    - type: nauc_recall_at_5_max
      value: 31.64779168744969
    - type: nauc_recall_at_5_std
      value: -2.1266696187991614
    - type: ndcg_at_1
      value: 27.571
    - type: ndcg_at_10
      value: 38.605000000000004
    - type: ndcg_at_100
      value: 44.069
    - type: ndcg_at_1000
      value: 46.01
    - type: ndcg_at_20
      value: 40.735
    - type: ndcg_at_3
      value: 33.928000000000004
    - type: ndcg_at_5
      value: 36.305
    - type: precision_at_1
      value: 27.571
    - type: precision_at_10
      value: 5.876
    - type: precision_at_100
      value: 0.905
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_20
      value: 3.4520000000000004
    - type: precision_at_3
      value: 14.388000000000002
    - type: precision_at_5
      value: 10.034
    - type: recall_at_1
      value: 25.635
    - type: recall_at_10
      value: 51.095
    - type: recall_at_100
      value: 76.369
    - type: recall_at_1000
      value: 90.724
    - type: recall_at_20
      value: 59.123000000000005
    - type: recall_at_3
      value: 38.45
    - type: recall_at_5
      value: 44.228
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackMathematicaRetrieval (default)
      revision: 90fceea13679c63fe563ded68f3b6f06e50061de
      split: test
      type: mteb/cqadupstack-mathematica
    metrics:
    - type: main_score
      value: 30.764999999999997
    - type: map_at_1
      value: 16.658
    - type: map_at_10
      value: 25.224000000000004
    - type: map_at_100
      value: 26.509
    - type: map_at_1000
      value: 26.632
    - type: map_at_20
      value: 25.913000000000004
    - type: map_at_3
      value: 22.601
    - type: map_at_5
      value: 23.967
    - type: mrr_at_1
      value: 21.144278606965177
    - type: mrr_at_10
      value: 30.181137960988707
    - type: mrr_at_100
      value: 31.228740132951536
    - type: mrr_at_1000
      value: 31.29708310421508
    - type: mrr_at_20
      value: 30.830392992680434
    - type: mrr_at_3
      value: 27.736318407960198
    - type: mrr_at_5
      value: 29.079601990049753
    - type: nauc_map_at_1000_diff1
      value: 28.215784470844028
    - type: nauc_map_at_1000_max
      value: 15.11609731383213
    - type: nauc_map_at_1000_std
      value: 0.3574750865464488
    - type: nauc_map_at_100_diff1
      value: 28.25047984624306
    - type: nauc_map_at_100_max
      value: 15.127062741715122
    - type: nauc_map_at_100_std
      value: 0.3697567511363066
    - type: nauc_map_at_10_diff1
      value: 28.427181607714058
    - type: nauc_map_at_10_max
      value: 15.12872402474189
    - type: nauc_map_at_10_std
      value: -0.2241757315002717
    - type: nauc_map_at_1_diff1
      value: 33.937769610150795
    - type: nauc_map_at_1_max
      value: 14.995651613930416
    - type: nauc_map_at_1_std
      value: -0.6617940295904605
    - type: nauc_map_at_20_diff1
      value: 28.21006834119646
    - type: nauc_map_at_20_max
      value: 15.02744009126895
    - type: nauc_map_at_20_std
      value: 0.08823997857740566
    - type: nauc_map_at_3_diff1
      value: 29.173496560017416
    - type: nauc_map_at_3_max
      value: 14.639104189699575
    - type: nauc_map_at_3_std
      value: -0.6921945279466096
    - type: nauc_map_at_5_diff1
      value: 29.163514975981116
    - type: nauc_map_at_5_max
      value: 14.689551020960417
    - type: nauc_map_at_5_std
      value: -0.9884110088458646
    - type: nauc_mrr_at_1000_diff1
      value: 26.345565693524204
    - type: nauc_mrr_at_1000_max
      value: 15.347578473349024
    - type: nauc_mrr_at_1000_std
      value: -0.21516859018828183
    - type: nauc_mrr_at_100_diff1
      value: 26.354070228572102
    - type: nauc_mrr_at_100_max
      value: 15.367211905095218
    - type: nauc_mrr_at_100_std
      value: -0.21964805294692075
    - type: nauc_mrr_at_10_diff1
      value: 26.522830659969433
    - type: nauc_mrr_at_10_max
      value: 15.41462814452358
    - type: nauc_mrr_at_10_std
      value: -0.5668628458188272
    - type: nauc_mrr_at_1_diff1
      value: 31.402709811424927
    - type: nauc_mrr_at_1_max
      value: 14.418493060958776
    - type: nauc_mrr_at_1_std
      value: -1.6517926656745607
    - type: nauc_mrr_at_20_diff1
      value: 26.282456675522898
    - type: nauc_mrr_at_20_max
      value: 15.447241286988644
    - type: nauc_mrr_at_20_std
      value: -0.32626358373542236
    - type: nauc_mrr_at_3_diff1
      value: 27.546860687111497
    - type: nauc_mrr_at_3_max
      value: 14.865589228333079
    - type: nauc_mrr_at_3_std
      value: -1.3068709379305234
    - type: nauc_mrr_at_5_diff1
      value: 27.06509986242121
    - type: nauc_mrr_at_5_max
      value: 15.177763098002364
    - type: nauc_mrr_at_5_std
      value: -1.16930774522256
    - type: nauc_ndcg_at_1000_diff1
      value: 24.709090767755622
    - type: nauc_ndcg_at_1000_max
      value: 15.520090334565054
    - type: nauc_ndcg_at_1000_std
      value: 2.9533344862667747
    - type: nauc_ndcg_at_100_diff1
      value: 25.356157577233983
    - type: nauc_ndcg_at_100_max
      value: 16.079572827297824
    - type: nauc_ndcg_at_100_std
      value: 3.622693473618125
    - type: nauc_ndcg_at_10_diff1
      value: 25.955639154663636
    - type: nauc_ndcg_at_10_max
      value: 15.943425645624764
    - type: nauc_ndcg_at_10_std
      value: 0.5008584867513407
    - type: nauc_ndcg_at_1_diff1
      value: 31.402709811424927
    - type: nauc_ndcg_at_1_max
      value: 14.418493060958776
    - type: nauc_ndcg_at_1_std
      value: -1.6517926656745607
    - type: nauc_ndcg_at_20_diff1
      value: 25.095353897780015
    - type: nauc_ndcg_at_20_max
      value: 15.891462742861226
    - type: nauc_ndcg_at_20_std
      value: 1.6342009581966903
    - type: nauc_ndcg_at_3_diff1
      value: 27.656071042217018
    - type: nauc_ndcg_at_3_max
      value: 14.732664275411972
    - type: nauc_ndcg_at_3_std
      value: -1.049086998129956
    - type: nauc_ndcg_at_5_diff1
      value: 27.38503281404232
    - type: nauc_ndcg_at_5_max
      value: 15.034640286714454
    - type: nauc_ndcg_at_5_std
      value: -0.972027889449345
    - type: nauc_precision_at_1000_diff1
      value: -9.057547973066667
    - type: nauc_precision_at_1000_max
      value: 0.0379842398676796
    - type: nauc_precision_at_1000_std
      value: -2.3715955577838255
    - type: nauc_precision_at_100_diff1
      value: 3.435199797184808
    - type: nauc_precision_at_100_max
      value: 10.425155931620782
    - type: nauc_precision_at_100_std
      value: 6.872908216483914
    - type: nauc_precision_at_10_diff1
      value: 15.422351823373157
    - type: nauc_precision_at_10_max
      value: 16.34010838208429
    - type: nauc_precision_at_10_std
      value: -1.2753998058176044
    - type: nauc_precision_at_1_diff1
      value: 31.402709811424927
    - type: nauc_precision_at_1_max
      value: 14.418493060958776
    - type: nauc_precision_at_1_std
      value: -1.6517926656745607
    - type: nauc_precision_at_20_diff1
      value: 10.313837386639085
    - type: nauc_precision_at_20_max
      value: 13.78910257689541
    - type: nauc_precision_at_20_std
      value: 1.5929398493017257
    - type: nauc_precision_at_3_diff1
      value: 23.25696066460761
    - type: nauc_precision_at_3_max
      value: 14.955169627394794
    - type: nauc_precision_at_3_std
      value: -1.7693875268867107
    - type: nauc_precision_at_5_diff1
      value: 21.110217391295645
    - type: nauc_precision_at_5_max
      value: 14.940298857823638
    - type: nauc_precision_at_5_std
      value: -3.141179615732845
    - type: nauc_recall_at_1000_diff1
      value: -3.1975419332256156
    - type: nauc_recall_at_1000_max
      value: 10.709003792262061
    - type: nauc_recall_at_1000_std
      value: 29.632515436751344
    - type: nauc_recall_at_100_diff1
      value: 14.753573278728979
    - type: nauc_recall_at_100_max
      value: 16.734106566544742
    - type: nauc_recall_at_100_std
      value: 19.640918372019183
    - type: nauc_recall_at_10_diff1
      value: 18.909128770832908
    - type: nauc_recall_at_10_max
      value: 16.101028570442505
    - type: nauc_recall_at_10_std
      value: 2.802823990367418
    - type: nauc_recall_at_1_diff1
      value: 33.937769610150795
    - type: nauc_recall_at_1_max
      value: 14.995651613930416
    - type: nauc_recall_at_1_std
      value: -0.6617940295904605
    - type: nauc_recall_at_20_diff1
      value: 15.257690585135633
    - type: nauc_recall_at_20_max
      value: 16.115628620217766
    - type: nauc_recall_at_20_std
      value: 6.753295584561113
    - type: nauc_recall_at_3_diff1
      value: 23.926095750994747
    - type: nauc_recall_at_3_max
      value: 13.99845585410835
    - type: nauc_recall_at_3_std
      value: -0.22828473131289068
    - type: nauc_recall_at_5_diff1
      value: 23.005243803534924
    - type: nauc_recall_at_5_max
      value: 14.237328739082722
    - type: nauc_recall_at_5_std
      value: -0.41220770062972284
    - type: ndcg_at_1
      value: 21.144
    - type: ndcg_at_10
      value: 30.764999999999997
    - type: ndcg_at_100
      value: 36.796
    - type: ndcg_at_1000
      value: 39.528
    - type: ndcg_at_20
      value: 33.115
    - type: ndcg_at_3
      value: 25.935999999999996
    - type: ndcg_at_5
      value: 27.977
    - type: precision_at_1
      value: 21.144
    - type: precision_at_10
      value: 5.771
    - type: precision_at_100
      value: 1.005
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_20
      value: 3.526
    - type: precision_at_3
      value: 12.645000000000001
    - type: precision_at_5
      value: 9.154
    - type: recall_at_1
      value: 16.658
    - type: recall_at_10
      value: 42.989
    - type: recall_at_100
      value: 69.219
    - type: recall_at_1000
      value: 88.189
    - type: recall_at_20
      value: 51.471999999999994
    - type: recall_at_3
      value: 29.322
    - type: recall_at_5
      value: 34.559
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackPhysicsRetrieval (default)
      revision: 79531abbd1fb92d06c6d6315a0cbbbf5bb247ea4
      split: test
      type: mteb/cqadupstack-physics
    metrics:
    - type: main_score
      value: 49.266
    - type: map_at_1
      value: 31.746999999999996
    - type: map_at_10
      value: 43.074
    - type: map_at_100
      value: 44.330000000000005
    - type: map_at_1000
      value: 44.438
    - type: map_at_20
      value: 43.758
    - type: map_at_3
      value: 39.811
    - type: map_at_5
      value: 41.593999999999994
    - type: mrr_at_1
      value: 38.3060635226179
    - type: mrr_at_10
      value: 48.51115999816674
    - type: mrr_at_100
      value: 49.25975468215056
    - type: mrr_at_1000
      value: 49.29671341478951
    - type: mrr_at_20
      value: 48.91221683618144
    - type: mrr_at_3
      value: 45.957651588065445
    - type: mrr_at_5
      value: 47.569778633301254
    - type: nauc_map_at_1000_diff1
      value: 51.77460156668234
    - type: nauc_map_at_1000_max
      value: 32.72920421176147
    - type: nauc_map_at_1000_std
      value: -3.664986151133605
    - type: nauc_map_at_100_diff1
      value: 51.771275802162805
    - type: nauc_map_at_100_max
      value: 32.72744104484486
    - type: nauc_map_at_100_std
      value: -3.6823934082350607
    - type: nauc_map_at_10_diff1
      value: 51.79131105856843
    - type: nauc_map_at_10_max
      value: 32.506794049829736
    - type: nauc_map_at_10_std
      value: -4.319131139789252
    - type: nauc_map_at_1_diff1
      value: 58.36422244417587
    - type: nauc_map_at_1_max
      value: 31.044391176679774
    - type: nauc_map_at_1_std
      value: -6.816395166276562
    - type: nauc_map_at_20_diff1
      value: 51.68545332970695
    - type: nauc_map_at_20_max
      value: 32.63532445350996
    - type: nauc_map_at_20_std
      value: -4.012106748544064
    - type: nauc_map_at_3_diff1
      value: 52.383034989359075
    - type: nauc_map_at_3_max
      value: 32.33293757077801
    - type: nauc_map_at_3_std
      value: -5.22412984032333
    - type: nauc_map_at_5_diff1
      value: 51.626775210942256
    - type: nauc_map_at_5_max
      value: 32.481456607777986
    - type: nauc_map_at_5_std
      value: -4.910211384753943
    - type: nauc_mrr_at_1000_diff1
      value: 51.90348585647991
    - type: nauc_mrr_at_1000_max
      value: 33.83234530616177
    - type: nauc_mrr_at_1000_std
      value: -2.968775178475681
    - type: nauc_mrr_at_100_diff1
      value: 51.89090737763241
    - type: nauc_mrr_at_100_max
      value: 33.82998662563648
    - type: nauc_mrr_at_100_std
      value: -2.9679069204868993
    - type: nauc_mrr_at_10_diff1
      value: 51.754166308016934
    - type: nauc_mrr_at_10_max
      value: 33.671063503610505
    - type: nauc_mrr_at_10_std
      value: -3.161444910678319
    - type: nauc_mrr_at_1_diff1
      value: 58.01659131526815
    - type: nauc_mrr_at_1_max
      value: 33.929176058973475
    - type: nauc_mrr_at_1_std
      value: -4.285514101307509
    - type: nauc_mrr_at_20_diff1
      value: 51.77850807858162
    - type: nauc_mrr_at_20_max
      value: 33.8032664723237
    - type: nauc_mrr_at_20_std
      value: -3.0307887631291655
    - type: nauc_mrr_at_3_diff1
      value: 52.63517785475077
    - type: nauc_mrr_at_3_max
      value: 34.318283166548596
    - type: nauc_mrr_at_3_std
      value: -3.4963499060768766
    - type: nauc_mrr_at_5_diff1
      value: 51.65298151045795
    - type: nauc_mrr_at_5_max
      value: 33.78349197533825
    - type: nauc_mrr_at_5_std
      value: -3.3889198033346
    - type: nauc_ndcg_at_1000_diff1
      value: 49.938505950792056
    - type: nauc_ndcg_at_1000_max
      value: 33.312821868362626
    - type: nauc_ndcg_at_1000_std
      value: -1.2052475045661974
    - type: nauc_ndcg_at_100_diff1
      value: 49.61666657508662
    - type: nauc_ndcg_at_100_max
      value: 33.3065910878159
    - type: nauc_ndcg_at_100_std
      value: -0.8756090039879582
    - type: nauc_ndcg_at_10_diff1
      value: 49.44883192019154
    - type: nauc_ndcg_at_10_max
      value: 32.4393563444836
    - type: nauc_ndcg_at_10_std
      value: -2.892697727465909
    - type: nauc_ndcg_at_1_diff1
      value: 58.01659131526815
    - type: nauc_ndcg_at_1_max
      value: 33.929176058973475
    - type: nauc_ndcg_at_1_std
      value: -4.285514101307509
    - type: nauc_ndcg_at_20_diff1
      value: 49.071579605243
    - type: nauc_ndcg_at_20_max
      value: 32.86971964345875
    - type: nauc_ndcg_at_20_std
      value: -2.099003909275664
    - type: nauc_ndcg_at_3_diff1
      value: 50.96144919708894
    - type: nauc_ndcg_at_3_max
      value: 33.25540300473652
    - type: nauc_ndcg_at_3_std
      value: -3.9974431575303075
    - type: nauc_ndcg_at_5_diff1
      value: 49.49453089765371
    - type: nauc_ndcg_at_5_max
      value: 32.65956969036793
    - type: nauc_ndcg_at_5_std
      value: -3.793764206815637
    - type: nauc_precision_at_1000_diff1
      value: -16.58079415943088
    - type: nauc_precision_at_1000_max
      value: -4.97912447265626
    - type: nauc_precision_at_1000_std
      value: 10.224502329990587
    - type: nauc_precision_at_100_diff1
      value: -5.351552447335853
    - type: nauc_precision_at_100_max
      value: 6.409109513562239
    - type: nauc_precision_at_100_std
      value: 14.509573599570755
    - type: nauc_precision_at_10_diff1
      value: 14.318051235383916
    - type: nauc_precision_at_10_max
      value: 19.90256242270515
    - type: nauc_precision_at_10_std
      value: 7.028415354350305
    - type: nauc_precision_at_1_diff1
      value: 58.01659131526815
    - type: nauc_precision_at_1_max
      value: 33.929176058973475
    - type: nauc_precision_at_1_std
      value: -4.285514101307509
    - type: nauc_precision_at_20_diff1
      value: 7.209280496290979
    - type: nauc_precision_at_20_max
      value: 16.436808554496196
    - type: nauc_precision_at_20_std
      value: 10.427201704761409
    - type: nauc_precision_at_3_diff1
      value: 33.889963667569894
    - type: nauc_precision_at_3_max
      value: 31.189529983032088
    - type: nauc_precision_at_3_std
      value: 0.5896025293550511
    - type: nauc_precision_at_5_diff1
      value: 23.905124267124204
    - type: nauc_precision_at_5_max
      value: 27.015858736612238
    - type: nauc_precision_at_5_std
      value: 3.846364311192626
    - type: nauc_recall_at_1000_diff1
      value: 32.2096170080799
    - type: nauc_recall_at_1000_max
      value: 36.68204932322026
    - type: nauc_recall_at_1000_std
      value: 28.653820090746706
    - type: nauc_recall_at_100_diff1
      value: 34.837377699906284
    - type: nauc_recall_at_100_max
      value: 31.18392295582138
    - type: nauc_recall_at_100_std
      value: 14.884810582304835
    - type: nauc_recall_at_10_diff1
      value: 39.62246119933039
    - type: nauc_recall_at_10_max
      value: 28.517950097720618
    - type: nauc_recall_at_10_std
      value: 0.7547659273815758
    - type: nauc_recall_at_1_diff1
      value: 58.36422244417587
    - type: nauc_recall_at_1_max
      value: 31.044391176679774
    - type: nauc_recall_at_1_std
      value: -6.816395166276562
    - type: nauc_recall_at_20_diff1
      value: 36.38130914284682
    - type: nauc_recall_at_20_max
      value: 29.239309975521127
    - type: nauc_recall_at_20_std
      value: 3.981602105789861
    - type: nauc_recall_at_3_diff1
      value: 45.05319504295809
    - type: nauc_recall_at_3_max
      value: 31.073783740414886
    - type: nauc_recall_at_3_std
      value: -4.336705557625501
    - type: nauc_recall_at_5_diff1
      value: 40.46084158901563
    - type: nauc_recall_at_5_max
      value: 29.243788972075723
    - type: nauc_recall_at_5_std
      value: -3.4129528475658115
    - type: ndcg_at_1
      value: 38.306000000000004
    - type: ndcg_at_10
      value: 49.266
    - type: ndcg_at_100
      value: 54.31700000000001
    - type: ndcg_at_1000
      value: 56.211999999999996
    - type: ndcg_at_20
      value: 51.13999999999999
    - type: ndcg_at_3
      value: 44.147999999999996
    - type: ndcg_at_5
      value: 46.642
    - type: precision_at_1
      value: 38.306000000000004
    - type: precision_at_10
      value: 8.892999999999999
    - type: precision_at_100
      value: 1.327
    - type: precision_at_1000
      value: 0.168
    - type: precision_at_20
      value: 5.087
    - type: precision_at_3
      value: 20.821
    - type: precision_at_5
      value: 14.726
    - type: recall_at_1
      value: 31.746999999999996
    - type: recall_at_10
      value: 61.974
    - type: recall_at_100
      value: 82.875
    - type: recall_at_1000
      value: 95.004
    - type: recall_at_20
      value: 68.37100000000001
    - type: recall_at_3
      value: 47.964
    - type: recall_at_5
      value: 54.336
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackProgrammersRetrieval (default)
      revision: 6184bc1440d2dbc7612be22b50686b8826d22b32
      split: test
      type: mteb/cqadupstack-programmers
    metrics:
    - type: main_score
      value: 45.416000000000004
    - type: map_at_1
      value: 27.009
    - type: map_at_10
      value: 38.861000000000004
    - type: map_at_100
      value: 40.224
    - type: map_at_1000
      value: 40.314
    - type: map_at_20
      value: 39.612
    - type: map_at_3
      value: 35.247
    - type: map_at_5
      value: 37.175000000000004
    - type: mrr_at_1
      value: 33.67579908675799
    - type: mrr_at_10
      value: 44.02116402116402
    - type: mrr_at_100
      value: 44.888954015766146
    - type: mrr_at_1000
      value: 44.928917137747426
    - type: mrr_at_20
      value: 44.505673809993866
    - type: mrr_at_3
      value: 41.2861491628615
    - type: mrr_at_5
      value: 42.861491628614914
    - type: nauc_map_at_1000_diff1
      value: 44.79801952188031
    - type: nauc_map_at_1000_max
      value: 38.30483650881305
    - type: nauc_map_at_1000_std
      value: 0.6373883692777299
    - type: nauc_map_at_100_diff1
      value: 44.78791588055703
    - type: nauc_map_at_100_max
      value: 38.28702533941684
    - type: nauc_map_at_100_std
      value: 0.6242579432286671
    - type: nauc_map_at_10_diff1
      value: 44.84261493500037
    - type: nauc_map_at_10_max
      value: 37.96445073801114
    - type: nauc_map_at_10_std
      value: -0.10166688326399165
    - type: nauc_map_at_1_diff1
      value: 50.578505398127064
    - type: nauc_map_at_1_max
      value: 35.53594767391443
    - type: nauc_map_at_1_std
      value: -6.4643598361864525
    - type: nauc_map_at_20_diff1
      value: 44.77319531960517
    - type: nauc_map_at_20_max
      value: 38.049743697998586
    - type: nauc_map_at_20_std
      value: 0.287504416757325
    - type: nauc_map_at_3_diff1
      value: 45.43902179173615
    - type: nauc_map_at_3_max
      value: 37.392747827500486
    - type: nauc_map_at_3_std
      value: -1.6969063381604679
    - type: nauc_map_at_5_diff1
      value: 45.23947679166256
    - type: nauc_map_at_5_max
      value: 37.51395435521589
    - type: nauc_map_at_5_std
      value: -1.101677215971638
    - type: nauc_mrr_at_1000_diff1
      value: 44.30064825347452
    - type: nauc_mrr_at_1000_max
      value: 41.35577174791693
    - type: nauc_mrr_at_1000_std
      value: 3.016409335551971
    - type: nauc_mrr_at_100_diff1
      value: 44.288624013167706
    - type: nauc_mrr_at_100_max
      value: 41.37658471603471
    - type: nauc_mrr_at_100_std
      value: 3.0574297142086158
    - type: nauc_mrr_at_10_diff1
      value: 44.08660856211678
    - type: nauc_mrr_at_10_max
      value: 41.20765868269297
    - type: nauc_mrr_at_10_std
      value: 2.706951772565681
    - type: nauc_mrr_at_1_diff1
      value: 49.408134080092914
    - type: nauc_mrr_at_1_max
      value: 41.36353781796639
    - type: nauc_mrr_at_1_std
      value: -2.630064401284397
    - type: nauc_mrr_at_20_diff1
      value: 44.24307684262044
    - type: nauc_mrr_at_20_max
      value: 41.27080857366453
    - type: nauc_mrr_at_20_std
      value: 3.0209311276189332
    - type: nauc_mrr_at_3_diff1
      value: 44.67883635289191
    - type: nauc_mrr_at_3_max
      value: 41.54974426367144
    - type: nauc_mrr_at_3_std
      value: 2.242664415996028
    - type: nauc_mrr_at_5_diff1
      value: 44.28793388593288
    - type: nauc_mrr_at_5_max
      value: 40.82311380994332
    - type: nauc_mrr_at_5_std
      value: 2.082279477569418
    - type: nauc_ndcg_at_1000_diff1
      value: 43.04506086061929
    - type: nauc_ndcg_at_1000_max
      value: 40.144722059317075
    - type: nauc_ndcg_at_1000_std
      value: 5.418872303533302
    - type: nauc_ndcg_at_100_diff1
      value: 42.8517043314021
    - type: nauc_ndcg_at_100_max
      value: 40.21084483588522
    - type: nauc_ndcg_at_100_std
      value: 5.950431722018279
    - type: nauc_ndcg_at_10_diff1
      value: 42.7783457430856
    - type: nauc_ndcg_at_10_max
      value: 38.73997067642053
    - type: nauc_ndcg_at_10_std
      value: 3.1786315676919266
    - type: nauc_ndcg_at_1_diff1
      value: 49.408134080092914
    - type: nauc_ndcg_at_1_max
      value: 41.36353781796639
    - type: nauc_ndcg_at_1_std
      value: -2.630064401284397
    - type: nauc_ndcg_at_20_diff1
      value: 42.82651998987242
    - type: nauc_ndcg_at_20_max
      value: 38.95106287711829
    - type: nauc_ndcg_at_20_std
      value: 4.450192890715282
    - type: nauc_ndcg_at_3_diff1
      value: 43.3961642126806
    - type: nauc_ndcg_at_3_max
      value: 38.629041351775356
    - type: nauc_ndcg_at_3_std
      value: 1.514193580623193
    - type: nauc_ndcg_at_5_diff1
      value: 43.24986651709832
    - type: nauc_ndcg_at_5_max
      value: 37.875744530244674
    - type: nauc_ndcg_at_5_std
      value: 1.3451708281818235
    - type: nauc_precision_at_1000_diff1
      value: -2.8125605993501392
    - type: nauc_precision_at_1000_max
      value: 5.877093364297313
    - type: nauc_precision_at_1000_std
      value: 13.488509460301989
    - type: nauc_precision_at_100_diff1
      value: 0.8100433271243155
    - type: nauc_precision_at_100_max
      value: 16.14937197481945
    - type: nauc_precision_at_100_std
      value: 20.466564331204243
    - type: nauc_precision_at_10_diff1
      value: 17.071659321387173
    - type: nauc_precision_at_10_max
      value: 29.88600515447819
    - type: nauc_precision_at_10_std
      value: 16.02161631767335
    - type: nauc_precision_at_1_diff1
      value: 49.408134080092914
    - type: nauc_precision_at_1_max
      value: 41.36353781796639
    - type: nauc_precision_at_1_std
      value: -2.630064401284397
    - type: nauc_precision_at_20_diff1
      value: 10.335233341033216
    - type: nauc_precision_at_20_max
      value: 24.406112227026266
    - type: nauc_precision_at_20_std
      value: 18.629042101979017
    - type: nauc_precision_at_3_diff1
      value: 30.295117995390353
    - type: nauc_precision_at_3_max
      value: 37.18797144366785
    - type: nauc_precision_at_3_std
      value: 10.218510427277732
    - type: nauc_precision_at_5_diff1
      value: 25.18863246940531
    - type: nauc_precision_at_5_max
      value: 33.741715231576094
    - type: nauc_precision_at_5_std
      value: 11.654926220669878
    - type: nauc_recall_at_1000_diff1
      value: 25.86237249322086
    - type: nauc_recall_at_1000_max
      value: 53.17803351374376
    - type: nauc_recall_at_1000_std
      value: 63.217841320163004
    - type: nauc_recall_at_100_diff1
      value: 30.859977609271073
    - type: nauc_recall_at_100_max
      value: 44.05193026154527
    - type: nauc_recall_at_100_std
      value: 34.55223443238623
    - type: nauc_recall_at_10_diff1
      value: 34.187417376599335
    - type: nauc_recall_at_10_max
      value: 34.899128014812625
    - type: nauc_recall_at_10_std
      value: 9.671042355840063
    - type: nauc_recall_at_1_diff1
      value: 50.578505398127064
    - type: nauc_recall_at_1_max
      value: 35.53594767391443
    - type: nauc_recall_at_1_std
      value: -6.4643598361864525
    - type: nauc_recall_at_20_diff1
      value: 34.10741069308342
    - type: nauc_recall_at_20_max
      value: 35.4611933907094
    - type: nauc_recall_at_20_std
      value: 16.042290846069037
    - type: nauc_recall_at_3_diff1
      value: 38.375770341834766
    - type: nauc_recall_at_3_max
      value: 35.18604542713948
    - type: nauc_recall_at_3_std
      value: 3.0439931673585314
    - type: nauc_recall_at_5_diff1
      value: 36.58712538006672
    - type: nauc_recall_at_5_max
      value: 33.03823430863331
    - type: nauc_recall_at_5_std
      value: 3.5465788170015506
    - type: ndcg_at_1
      value: 33.676
    - type: ndcg_at_10
      value: 45.416000000000004
    - type: ndcg_at_100
      value: 51.068999999999996
    - type: ndcg_at_1000
      value: 52.734
    - type: ndcg_at_20
      value: 47.606
    - type: ndcg_at_3
      value: 39.749
    - type: ndcg_at_5
      value: 42.224000000000004
    - type: precision_at_1
      value: 33.676
    - type: precision_at_10
      value: 8.584
    - type: precision_at_100
      value: 1.331
    - type: precision_at_1000
      value: 0.164
    - type: precision_at_20
      value: 5.011
    - type: precision_at_3
      value: 19.521
    - type: precision_at_5
      value: 13.904
    - type: recall_at_1
      value: 27.009
    - type: recall_at_10
      value: 59.653
    - type: recall_at_100
      value: 83.612
    - type: recall_at_1000
      value: 94.44900000000001
    - type: recall_at_20
      value: 67.536
    - type: recall_at_3
      value: 43.614999999999995
    - type: recall_at_5
      value: 50.388999999999996
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackRetrieval (default)
      revision: CQADupstackRetrieval_is_a_combined_dataset
      split: test
      type: CQADupstackRetrieval_is_a_combined_dataset
    metrics:
    - type: main_score
      value: 42.70158333333334
    - type: ndcg_at_10
      value: 42.70158333333334
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackStatsRetrieval (default)
      revision: 65ac3a16b8e91f9cee4c9828cc7c335575432a2a
      split: test
      type: mteb/cqadupstack-stats
    metrics:
    - type: main_score
      value: 36.592
    - type: map_at_1
      value: 25.345000000000002
    - type: map_at_10
      value: 32.363
    - type: map_at_100
      value: 33.335
    - type: map_at_1000
      value: 33.433
    - type: map_at_20
      value: 32.878
    - type: map_at_3
      value: 30.233999999999998
    - type: map_at_5
      value: 31.198999999999998
    - type: mrr_at_1
      value: 28.37423312883436
    - type: mrr_at_10
      value: 35.011563930275585
    - type: mrr_at_100
      value: 35.85493824836966
    - type: mrr_at_1000
      value: 35.924045817604636
    - type: mrr_at_20
      value: 35.4727700542792
    - type: mrr_at_3
      value: 33.0521472392638
    - type: mrr_at_5
      value: 33.91104294478527
    - type: nauc_map_at_1000_diff1
      value: 52.10221090792941
    - type: nauc_map_at_1000_max
      value: 37.97870145738844
    - type: nauc_map_at_1000_std
      value: 4.434712869623427
    - type: nauc_map_at_100_diff1
      value: 52.04633689276877
    - type: nauc_map_at_100_max
      value: 37.922801925228335
    - type: nauc_map_at_100_std
      value: 4.407307285934232
    - type: nauc_map_at_10_diff1
      value: 52.311047625949726
    - type: nauc_map_at_10_max
      value: 37.693402148689344
    - type: nauc_map_at_10_std
      value: 4.150061933266761
    - type: nauc_map_at_1_diff1
      value: 59.40642930820075
    - type: nauc_map_at_1_max
      value: 38.11565621075943
    - type: nauc_map_at_1_std
      value: 0.2236602280183978
    - type: nauc_map_at_20_diff1
      value: 52.15382306116363
    - type: nauc_map_at_20_max
      value: 37.85709674777133
    - type: nauc_map_at_20_std
      value: 4.129360484091663
    - type: nauc_map_at_3_diff1
      value: 53.600964631491124
    - type: nauc_map_at_3_max
      value: 37.67276735065246
    - type: nauc_map_at_3_std
      value: 3.1559089879184805
    - type: nauc_map_at_5_diff1
      value: 52.6761419075841
    - type: nauc_map_at_5_max
      value: 37.67403489168251
    - type: nauc_map_at_5_std
      value: 4.0957976308400745
    - type: nauc_mrr_at_1000_diff1
      value: 51.894133976551934
    - type: nauc_mrr_at_1000_max
      value: 37.69150090605279
    - type: nauc_mrr_at_1000_std
      value: 5.53167090138349
    - type: nauc_mrr_at_100_diff1
      value: 51.86723432410493
    - type: nauc_mrr_at_100_max
      value: 37.675799119222056
    - type: nauc_mrr_at_100_std
      value: 5.525988076577769
    - type: nauc_mrr_at_10_diff1
      value: 52.069351041943555
    - type: nauc_mrr_at_10_max
      value: 37.58369095550716
    - type: nauc_mrr_at_10_std
      value: 5.411523585873688
    - type: nauc_mrr_at_1_diff1
      value: 59.9924403198651
    - type: nauc_mrr_at_1_max
      value: 38.60523574823761
    - type: nauc_mrr_at_1_std
      value: 2.920326325813509
    - type: nauc_mrr_at_20_diff1
      value: 52.019959211630194
    - type: nauc_mrr_at_20_max
      value: 37.68724731749795
    - type: nauc_mrr_at_20_std
      value: 5.284842719572277
    - type: nauc_mrr_at_3_diff1
      value: 53.15420839584364
    - type: nauc_mrr_at_3_max
      value: 37.638125057657994
    - type: nauc_mrr_at_3_std
      value: 5.070617161674469
    - type: nauc_mrr_at_5_diff1
      value: 52.091638315676214
    - type: nauc_mrr_at_5_max
      value: 37.453254508262496
    - type: nauc_mrr_at_5_std
      value: 5.437885115048488
    - type: nauc_ndcg_at_1000_diff1
      value: 48.93146473391044
    - type: nauc_ndcg_at_1000_max
      value: 38.66354073398188
    - type: nauc_ndcg_at_1000_std
      value: 7.864040979702623
    - type: nauc_ndcg_at_100_diff1
      value: 47.93516146668464
    - type: nauc_ndcg_at_100_max
      value: 37.96305590532788
    - type: nauc_ndcg_at_100_std
      value: 7.533630969864932
    - type: nauc_ndcg_at_10_diff1
      value: 49.314412298706586
    - type: nauc_ndcg_at_10_max
      value: 37.14853880579557
    - type: nauc_ndcg_at_10_std
      value: 5.788877196723973
    - type: nauc_ndcg_at_1_diff1
      value: 59.9924403198651
    - type: nauc_ndcg_at_1_max
      value: 38.60523574823761
    - type: nauc_ndcg_at_1_std
      value: 2.920326325813509
    - type: nauc_ndcg_at_20_diff1
      value: 48.97613498555538
    - type: nauc_ndcg_at_20_max
      value: 37.774155885053
    - type: nauc_ndcg_at_20_std
      value: 5.620396462179348
    - type: nauc_ndcg_at_3_diff1
      value: 51.339291445204324
    - type: nauc_ndcg_at_3_max
      value: 37.1134520118515
    - type: nauc_ndcg_at_3_std
      value: 4.65811198536688
    - type: nauc_ndcg_at_5_diff1
      value: 49.82566697465264
    - type: nauc_ndcg_at_5_max
      value: 37.074306731374556
    - type: nauc_ndcg_at_5_std
      value: 5.664342593293808
    - type: nauc_precision_at_1000_diff1
      value: 5.949568112100649
    - type: nauc_precision_at_1000_max
      value: 14.27613117400501
    - type: nauc_precision_at_1000_std
      value: 13.216661590735033
    - type: nauc_precision_at_100_diff1
      value: 15.949740304487781
    - type: nauc_precision_at_100_max
      value: 23.204110568209842
    - type: nauc_precision_at_100_std
      value: 16.778669748720645
    - type: nauc_precision_at_10_diff1
      value: 32.54460913253613
    - type: nauc_precision_at_10_max
      value: 30.077881263835422
    - type: nauc_precision_at_10_std
      value: 13.16689153718017
    - type: nauc_precision_at_1_diff1
      value: 59.9924403198651
    - type: nauc_precision_at_1_max
      value: 38.60523574823761
    - type: nauc_precision_at_1_std
      value: 2.920326325813509
    - type: nauc_precision_at_20_diff1
      value: 27.373037654570854
    - type: nauc_precision_at_20_max
      value: 29.036294626453635
    - type: nauc_precision_at_20_std
      value: 12.527255335691267
    - type: nauc_precision_at_3_diff1
      value: 43.324626111255085
    - type: nauc_precision_at_3_max
      value: 33.820314747164964
    - type: nauc_precision_at_3_std
      value: 9.606689063345634
    - type: nauc_precision_at_5_diff1
      value: 36.56979260506234
    - type: nauc_precision_at_5_max
      value: 32.104078603478584
    - type: nauc_precision_at_5_std
      value: 13.288924424488219
    - type: nauc_recall_at_1000_diff1
      value: 24.830155536816743
    - type: nauc_recall_at_1000_max
      value: 44.56989928611184
    - type: nauc_recall_at_1000_std
      value: 38.634755509856575
    - type: nauc_recall_at_100_diff1
      value: 26.54828559499917
    - type: nauc_recall_at_100_max
      value: 35.970249595603754
    - type: nauc_recall_at_100_std
      value: 21.178635451661513
    - type: nauc_recall_at_10_diff1
      value: 39.942357261790264
    - type: nauc_recall_at_10_max
      value: 34.64751717405597
    - type: nauc_recall_at_10_std
      value: 9.05242633651784
    - type: nauc_recall_at_1_diff1
      value: 59.40642930820075
    - type: nauc_recall_at_1_max
      value: 38.11565621075943
    - type: nauc_recall_at_1_std
      value: 0.2236602280183978
    - type: nauc_recall_at_20_diff1
      value: 37.28076595566659
    - type: nauc_recall_at_20_max
      value: 36.004705228338054
    - type: nauc_recall_at_20_std
      value: 8.013557071650954
    - type: nauc_recall_at_3_diff1
      value: 45.55894807036549
    - type: nauc_recall_at_3_max
      value: 35.14623776264798
    - type: nauc_recall_at_3_std
      value: 6.173033074247223
    - type: nauc_recall_at_5_diff1
      value: 41.985621749955385
    - type: nauc_recall_at_5_max
      value: 34.90602485658553
    - type: nauc_recall_at_5_std
      value: 8.545282928891847
    - type: ndcg_at_1
      value: 28.374
    - type: ndcg_at_10
      value: 36.592
    - type: ndcg_at_100
      value: 41.475
    - type: ndcg_at_1000
      value: 43.791999999999994
    - type: ndcg_at_20
      value: 38.321
    - type: ndcg_at_3
      value: 32.54
    - type: ndcg_at_5
      value: 33.994
    - type: precision_at_1
      value: 28.374
    - type: precision_at_10
      value: 5.736
    - type: precision_at_100
      value: 0.89
    - type: precision_at_1000
      value: 0.116
    - type: precision_at_20
      value: 3.321
    - type: precision_at_3
      value: 13.855
    - type: precision_at_5
      value: 9.447999999999999
    - type: recall_at_1
      value: 25.345000000000002
    - type: recall_at_10
      value: 46.961999999999996
    - type: recall_at_100
      value: 69.612
    - type: recall_at_1000
      value: 86.446
    - type: recall_at_20
      value: 53.43
    - type: recall_at_3
      value: 35.554
    - type: recall_at_5
      value: 39.117000000000004
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackTexRetrieval (default)
      revision: 46989137a86843e03a6195de44b09deda022eec7
      split: test
      type: mteb/cqadupstack-tex
    metrics:
    - type: main_score
      value: 30.59
    - type: map_at_1
      value: 17.867
    - type: map_at_10
      value: 25.576
    - type: map_at_100
      value: 26.672
    - type: map_at_1000
      value: 26.795
    - type: map_at_20
      value: 26.135
    - type: map_at_3
      value: 22.914
    - type: map_at_5
      value: 24.386
    - type: mrr_at_1
      value: 21.88575361321404
    - type: mrr_at_10
      value: 29.62001168900687
    - type: mrr_at_100
      value: 30.51188915651043
    - type: mrr_at_1000
      value: 30.586970551824294
    - type: mrr_at_20
      value: 30.099858193762604
    - type: mrr_at_3
      value: 27.185134205092908
    - type: mrr_at_5
      value: 28.537508602890572
    - type: nauc_map_at_1000_diff1
      value: 34.49325545158564
    - type: nauc_map_at_1000_max
      value: 27.556328790182178
    - type: nauc_map_at_1000_std
      value: -1.1473434205027908
    - type: nauc_map_at_100_diff1
      value: 34.502739591647064
    - type: nauc_map_at_100_max
      value: 27.543065656635115
    - type: nauc_map_at_100_std
      value: -1.155746617233167
    - type: nauc_map_at_10_diff1
      value: 34.72548353792797
    - type: nauc_map_at_10_max
      value: 27.458789010336247
    - type: nauc_map_at_10_std
      value: -1.5714147132657361
    - type: nauc_map_at_1_diff1
      value: 42.853500192581784
    - type: nauc_map_at_1_max
      value: 27.708121089387067
    - type: nauc_map_at_1_std
      value: -3.5799317178944947
    - type: nauc_map_at_20_diff1
      value: 34.54145190942222
    - type: nauc_map_at_20_max
      value: 27.46586240444109
    - type: nauc_map_at_20_std
      value: -1.5030706778324936
    - type: nauc_map_at_3_diff1
      value: 35.988304209191185
    - type: nauc_map_at_3_max
      value: 27.552309826648937
    - type: nauc_map_at_3_std
      value: -2.1955887759374986
    - type: nauc_map_at_5_diff1
      value: 35.2173583442156
    - type: nauc_map_at_5_max
      value: 27.412853618385153
    - type: nauc_map_at_5_std
      value: -1.9029886360661645
    - type: nauc_mrr_at_1000_diff1
      value: 33.55990759132534
    - type: nauc_mrr_at_1000_max
      value: 27.66214063434415
    - type: nauc_mrr_at_1000_std
      value: -1.2170382581999488
    - type: nauc_mrr_at_100_diff1
      value: 33.557973949776404
    - type: nauc_mrr_at_100_max
      value: 27.65525431001004
    - type: nauc_mrr_at_100_std
      value: -1.2064075673357395
    - type: nauc_mrr_at_10_diff1
      value: 33.66107975387345
    - type: nauc_mrr_at_10_max
      value: 27.649697817706954
    - type: nauc_mrr_at_10_std
      value: -1.3892620177060093
    - type: nauc_mrr_at_1_diff1
      value: 40.80816405416601
    - type: nauc_mrr_at_1_max
      value: 28.886179980607064
    - type: nauc_mrr_at_1_std
      value: -3.3951299182217025
    - type: nauc_mrr_at_20_diff1
      value: 33.478028161809625
    - type: nauc_mrr_at_20_max
      value: 27.573635545033493
    - type: nauc_mrr_at_20_std
      value: -1.3968331731988919
    - type: nauc_mrr_at_3_diff1
      value: 34.55895724248828
    - type: nauc_mrr_at_3_max
      value: 27.90796568552703
    - type: nauc_mrr_at_3_std
      value: -2.2006436714574154
    - type: nauc_mrr_at_5_diff1
      value: 34.028531995696525
    - type: nauc_mrr_at_5_max
      value: 27.66592068060555
    - type: nauc_mrr_at_5_std
      value: -1.7757292378537217
    - type: nauc_ndcg_at_1000_diff1
      value: 31.268481868963743
    - type: nauc_ndcg_at_1000_max
      value: 27.38872907691976
    - type: nauc_ndcg_at_1000_std
      value: 1.7346994429403535
    - type: nauc_ndcg_at_100_diff1
      value: 31.11876549838855
    - type: nauc_ndcg_at_100_max
      value: 27.278699810754976
    - type: nauc_ndcg_at_100_std
      value: 1.8155675026667304
    - type: nauc_ndcg_at_10_diff1
      value: 31.604755923595608
    - type: nauc_ndcg_at_10_max
      value: 27.08891042622456
    - type: nauc_ndcg_at_10_std
      value: -0.3767103827211869
    - type: nauc_ndcg_at_1_diff1
      value: 40.80816405416601
    - type: nauc_ndcg_at_1_max
      value: 28.886179980607064
    - type: nauc_ndcg_at_1_std
      value: -3.3951299182217025
    - type: nauc_ndcg_at_20_diff1
      value: 31.039555302570506
    - type: nauc_ndcg_at_20_max
      value: 26.84392449049619
    - type: nauc_ndcg_at_20_std
      value: -0.24222312435568985
    - type: nauc_ndcg_at_3_diff1
      value: 33.55360834070662
    - type: nauc_ndcg_at_3_max
      value: 27.43126257082683
    - type: nauc_ndcg_at_3_std
      value: -1.9241781351097627
    - type: nauc_ndcg_at_5_diff1
      value: 32.49208302212607
    - type: nauc_ndcg_at_5_max
      value: 27.198013350025917
    - type: nauc_ndcg_at_5_std
      value: -1.3092888773957871
    - type: nauc_precision_at_1000_diff1
      value: -7.18909274307431
    - type: nauc_precision_at_1000_max
      value: 8.589945404305544
    - type: nauc_precision_at_1000_std
      value: 7.146110108733285
    - type: nauc_precision_at_100_diff1
      value: 3.333505999348508
    - type: nauc_precision_at_100_max
      value: 16.475940897446037
    - type: nauc_precision_at_100_std
      value: 10.596147977162133
    - type: nauc_precision_at_10_diff1
      value: 16.593144782123474
    - type: nauc_precision_at_10_max
      value: 24.638834773962433
    - type: nauc_precision_at_10_std
      value: 2.47210009420964
    - type: nauc_precision_at_1_diff1
      value: 40.80816405416601
    - type: nauc_precision_at_1_max
      value: 28.886179980607064
    - type: nauc_precision_at_1_std
      value: -3.3951299182217025
    - type: nauc_precision_at_20_diff1
      value: 11.861028121381887
    - type: nauc_precision_at_20_max
      value: 22.08074759970412
    - type: nauc_precision_at_20_std
      value: 3.390946602078225
    - type: nauc_precision_at_3_diff1
      value: 25.23585309301632
    - type: nauc_precision_at_3_max
      value: 27.058413941233457
    - type: nauc_precision_at_3_std
      value: -1.490236891363537
    - type: nauc_precision_at_5_diff1
      value: 21.127003173889516
    - type: nauc_precision_at_5_max
      value: 25.651901791436433
    - type: nauc_precision_at_5_std
      value: -0.5404378139986246
    - type: nauc_recall_at_1000_diff1
      value: 11.918767165680597
    - type: nauc_recall_at_1000_max
      value: 24.163407046339934
    - type: nauc_recall_at_1000_std
      value: 26.198314738813927
    - type: nauc_recall_at_100_diff1
      value: 18.481964130791866
    - type: nauc_recall_at_100_max
      value: 23.95604920326788
    - type: nauc_recall_at_100_std
      value: 14.647767309162887
    - type: nauc_recall_at_10_diff1
      value: 23.045112570927536
    - type: nauc_recall_at_10_max
      value: 23.487123935917666
    - type: nauc_recall_at_10_std
      value: 2.649000895269065
    - type: nauc_recall_at_1_diff1
      value: 42.853500192581784
    - type: nauc_recall_at_1_max
      value: 27.708121089387067
    - type: nauc_recall_at_1_std
      value: -3.5799317178944947
    - type: nauc_recall_at_20_diff1
      value: 20.507448541593256
    - type: nauc_recall_at_20_max
      value: 22.315181357683365
    - type: nauc_recall_at_20_std
      value: 3.0119483062086156
    - type: nauc_recall_at_3_diff1
      value: 28.776376648840845
    - type: nauc_recall_at_3_max
      value: 25.51950461226441
    - type: nauc_recall_at_3_std
      value: -0.6820491549861496
    - type: nauc_recall_at_5_diff1
      value: 26.007225388693893
    - type: nauc_recall_at_5_max
      value: 24.470508824440525
    - type: nauc_recall_at_5_std
      value: 0.4541288136053105
    - type: ndcg_at_1
      value: 21.886
    - type: ndcg_at_10
      value: 30.59
    - type: ndcg_at_100
      value: 35.855
    - type: ndcg_at_1000
      value: 38.673
    - type: ndcg_at_20
      value: 32.385000000000005
    - type: ndcg_at_3
      value: 25.802000000000003
    - type: ndcg_at_5
      value: 28.022999999999996
    - type: precision_at_1
      value: 21.886
    - type: precision_at_10
      value: 5.592
    - type: precision_at_100
      value: 0.972
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_20
      value: 3.3619999999999997
    - type: precision_at_3
      value: 12.17
    - type: precision_at_5
      value: 8.906
    - type: recall_at_1
      value: 17.867
    - type: recall_at_10
      value: 41.864000000000004
    - type: recall_at_100
      value: 65.553
    - type: recall_at_1000
      value: 85.547
    - type: recall_at_20
      value: 48.379
    - type: recall_at_3
      value: 28.442
    - type: recall_at_5
      value: 34.14
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackUnixRetrieval (default)
      revision: 6c6430d3a6d36f8d2a829195bc5dc94d7e063e53
      split: test
      type: mteb/cqadupstack-unix
    metrics:
    - type: main_score
      value: 43.834
    - type: map_at_1
      value: 28.967
    - type: map_at_10
      value: 38.427
    - type: map_at_100
      value: 39.582
    - type: map_at_1000
      value: 39.678000000000004
    - type: map_at_20
      value: 39.035
    - type: map_at_3
      value: 35.83
    - type: map_at_5
      value: 37.108999999999995
    - type: mrr_at_1
      value: 33.3955223880597
    - type: mrr_at_10
      value: 42.31202617863066
    - type: mrr_at_100
      value: 43.21760494076947
    - type: mrr_at_1000
      value: 43.26900265192256
    - type: mrr_at_20
      value: 42.87328346103501
    - type: mrr_at_3
      value: 40.15858208955223
    - type: mrr_at_5
      value: 41.25466417910448
    - type: nauc_map_at_1000_diff1
      value: 53.048724616609334
    - type: nauc_map_at_1000_max
      value: 38.59286957068655
    - type: nauc_map_at_1000_std
      value: -3.512174792820818
    - type: nauc_map_at_100_diff1
      value: 53.05707527562925
    - type: nauc_map_at_100_max
      value: 38.595046358303655
    - type: nauc_map_at_100_std
      value: -3.5537585876182867
    - type: nauc_map_at_10_diff1
      value: 53.15959505177301
    - type: nauc_map_at_10_max
      value: 38.58423657698431
    - type: nauc_map_at_10_std
      value: -3.5669406976493576
    - type: nauc_map_at_1_diff1
      value: 59.55714192814266
    - type: nauc_map_at_1_max
      value: 38.14703046465308
    - type: nauc_map_at_1_std
      value: -4.898963990983661
    - type: nauc_map_at_20_diff1
      value: 53.06026987763403
    - type: nauc_map_at_20_max
      value: 38.64655863281934
    - type: nauc_map_at_20_std
      value: -3.5402964228130123
    - type: nauc_map_at_3_diff1
      value: 53.745507769177756
    - type: nauc_map_at_3_max
      value: 38.33821555664789
    - type: nauc_map_at_3_std
      value: -5.196841373466233
    - type: nauc_map_at_5_diff1
      value: 53.13844967494078
    - type: nauc_map_at_5_max
      value: 38.06711106859648
    - type: nauc_map_at_5_std
      value: -4.202472757208345
    - type: nauc_mrr_at_1000_diff1
      value: 51.65346161355801
    - type: nauc_mrr_at_1000_max
      value: 38.33149378133196
    - type: nauc_mrr_at_1000_std
      value: -3.072867480769094
    - type: nauc_mrr_at_100_diff1
      value: 51.63874925422337
    - type: nauc_mrr_at_100_max
      value: 38.314456402931036
    - type: nauc_mrr_at_100_std
      value: -3.1137837675138904
    - type: nauc_mrr_at_10_diff1
      value: 51.545157392866955
    - type: nauc_mrr_at_10_max
      value: 38.40980320087051
    - type: nauc_mrr_at_10_std
      value: -2.9626720726283566
    - type: nauc_mrr_at_1_diff1
      value: 58.89662693218527
    - type: nauc_mrr_at_1_max
      value: 39.9940049100829
    - type: nauc_mrr_at_1_std
      value: -4.644526276718296
    - type: nauc_mrr_at_20_diff1
      value: 51.496137946752505
    - type: nauc_mrr_at_20_max
      value: 38.302574049764324
    - type: nauc_mrr_at_20_std
      value: -3.102766458691138
    - type: nauc_mrr_at_3_diff1
      value: 51.933389191645176
    - type: nauc_mrr_at_3_max
      value: 38.376004190745235
    - type: nauc_mrr_at_3_std
      value: -4.407227427794424
    - type: nauc_mrr_at_5_diff1
      value: 51.43121618223506
    - type: nauc_mrr_at_5_max
      value: 37.799347167768815
    - type: nauc_mrr_at_5_std
      value: -3.6783828270522823
    - type: nauc_ndcg_at_1000_diff1
      value: 50.65372433977417
    - type: nauc_ndcg_at_1000_max
      value: 38.76438055840659
    - type: nauc_ndcg_at_1000_std
      value: -1.3042509316937945
    - type: nauc_ndcg_at_100_diff1
      value: 50.75111932365467
    - type: nauc_ndcg_at_100_max
      value: 38.63655288892062
    - type: nauc_ndcg_at_100_std
      value: -1.9054159551550338
    - type: nauc_ndcg_at_10_diff1
      value: 50.60080476375649
    - type: nauc_ndcg_at_10_max
      value: 38.793151280176
    - type: nauc_ndcg_at_10_std
      value: -1.9283076169081044
    - type: nauc_ndcg_at_1_diff1
      value: 58.89662693218527
    - type: nauc_ndcg_at_1_max
      value: 39.9940049100829
    - type: nauc_ndcg_at_1_std
      value: -4.644526276718296
    - type: nauc_ndcg_at_20_diff1
      value: 50.230064043657364
    - type: nauc_ndcg_at_20_max
      value: 38.891161104508384
    - type: nauc_ndcg_at_20_std
      value: -1.8913841784874814
    - type: nauc_ndcg_at_3_diff1
      value: 51.17438951964256
    - type: nauc_ndcg_at_3_max
      value: 38.28206207948888
    - type: nauc_ndcg_at_3_std
      value: -5.088534709798436
    - type: nauc_ndcg_at_5_diff1
      value: 50.489004449358376
    - type: nauc_ndcg_at_5_max
      value: 37.43922192422311
    - type: nauc_ndcg_at_5_std
      value: -3.6176744423900744
    - type: nauc_precision_at_1000_diff1
      value: -15.999713051406678
    - type: nauc_precision_at_1000_max
      value: -6.513916213760548
    - type: nauc_precision_at_1000_std
      value: 1.0280197325766425
    - type: nauc_precision_at_100_diff1
      value: -0.71946065269529
    - type: nauc_precision_at_100_max
      value: 7.748000953864443
    - type: nauc_precision_at_100_std
      value: 2.8208123480463554
    - type: nauc_precision_at_10_diff1
      value: 22.810537271127078
    - type: nauc_precision_at_10_max
      value: 28.755965074901457
    - type: nauc_precision_at_10_std
      value: 3.7100114329436766
    - type: nauc_precision_at_1_diff1
      value: 58.89662693218527
    - type: nauc_precision_at_1_max
      value: 39.9940049100829
    - type: nauc_precision_at_1_std
      value: -4.644526276718296
    - type: nauc_precision_at_20_diff1
      value: 14.893675551571434
    - type: nauc_precision_at_20_max
      value: 23.193110884243882
    - type: nauc_precision_at_20_std
      value: 3.6389553264417387
    - type: nauc_precision_at_3_diff1
      value: 37.57595902015152
    - type: nauc_precision_at_3_max
      value: 35.11741099174917
    - type: nauc_precision_at_3_std
      value: -5.184422279738604
    - type: nauc_precision_at_5_diff1
      value: 31.262880019582873
    - type: nauc_precision_at_5_max
      value: 31.14051710549249
    - type: nauc_precision_at_5_std
      value: -0.8630132957747255
    - type: nauc_recall_at_1000_diff1
      value: 33.12291384838218
    - type: nauc_recall_at_1000_max
      value: 48.1548592613286
    - type: nauc_recall_at_1000_std
      value: 36.59889023810872
    - type: nauc_recall_at_100_diff1
      value: 42.63909611714733
    - type: nauc_recall_at_100_max
      value: 37.67651410866005
    - type: nauc_recall_at_100_std
      value: 6.143735639048131
    - type: nauc_recall_at_10_diff1
      value: 42.90408746336859
    - type: nauc_recall_at_10_max
      value: 38.12496862811442
    - type: nauc_recall_at_10_std
      value: 2.714597237277699
    - type: nauc_recall_at_1_diff1
      value: 59.55714192814266
    - type: nauc_recall_at_1_max
      value: 38.14703046465308
    - type: nauc_recall_at_1_std
      value: -4.898963990983661
    - type: nauc_recall_at_20_diff1
      value: 40.43198203016414
    - type: nauc_recall_at_20_max
      value: 38.658149745252054
    - type: nauc_recall_at_20_std
      value: 3.4823770242734806
    - type: nauc_recall_at_3_diff1
      value: 46.02442854627347
    - type: nauc_recall_at_3_max
      value: 36.45635273472127
    - type: nauc_recall_at_3_std
      value: -5.50129428564403
    - type: nauc_recall_at_5_diff1
      value: 43.35611512922718
    - type: nauc_recall_at_5_max
      value: 34.62150709194741
    - type: nauc_recall_at_5_std
      value: -2.280954879243513
    - type: ndcg_at_1
      value: 33.396
    - type: ndcg_at_10
      value: 43.834
    - type: ndcg_at_100
      value: 49.152
    - type: ndcg_at_1000
      value: 51.185
    - type: ndcg_at_20
      value: 45.911
    - type: ndcg_at_3
      value: 39.201
    - type: ndcg_at_5
      value: 40.999
    - type: precision_at_1
      value: 33.396
    - type: precision_at_10
      value: 7.303999999999999
    - type: precision_at_100
      value: 1.1280000000000001
    - type: precision_at_1000
      value: 0.14100000000000001
    - type: precision_at_20
      value: 4.268000000000001
    - type: precision_at_3
      value: 17.693
    - type: precision_at_5
      value: 12.071
    - type: recall_at_1
      value: 28.967
    - type: recall_at_10
      value: 56.338
    - type: recall_at_100
      value: 79.074
    - type: recall_at_1000
      value: 93.028
    - type: recall_at_20
      value: 63.709
    - type: recall_at_3
      value: 43.422
    - type: recall_at_5
      value: 48.082
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackWebmastersRetrieval (default)
      revision: 160c094312a0e1facb97e55eeddb698c0abe3571
      split: test
      type: mteb/cqadupstack-webmasters
    metrics:
    - type: main_score
      value: 42.276
    - type: map_at_1
      value: 27.913
    - type: map_at_10
      value: 36.85
    - type: map_at_100
      value: 38.677
    - type: map_at_1000
      value: 38.885999999999996
    - type: map_at_20
      value: 37.82
    - type: map_at_3
      value: 34.305
    - type: map_at_5
      value: 35.555
    - type: mrr_at_1
      value: 33.794466403162055
    - type: mrr_at_10
      value: 41.494761277369975
    - type: mrr_at_100
      value: 42.632112752966044
    - type: mrr_at_1000
      value: 42.676877714015085
    - type: mrr_at_20
      value: 42.19759601158817
    - type: mrr_at_3
      value: 39.26218708827404
    - type: mrr_at_5
      value: 40.5368906455863
    - type: nauc_map_at_1000_diff1
      value: 50.854865033167066
    - type: nauc_map_at_1000_max
      value: 27.577581890907393
    - type: nauc_map_at_1000_std
      value: 5.852234537571572
    - type: nauc_map_at_100_diff1
      value: 50.89518111760981
    - type: nauc_map_at_100_max
      value: 27.729183522675942
    - type: nauc_map_at_100_std
      value: 5.627818713948203
    - type: nauc_map_at_10_diff1
      value: 50.52622901234379
    - type: nauc_map_at_10_max
      value: 27.5061520637879
    - type: nauc_map_at_10_std
      value: 3.8309582134509474
    - type: nauc_map_at_1_diff1
      value: 54.85652943645487
    - type: nauc_map_at_1_max
      value: 24.50498114129174
    - type: nauc_map_at_1_std
      value: 0.3799552038716688
    - type: nauc_map_at_20_diff1
      value: 50.78996825989718
    - type: nauc_map_at_20_max
      value: 27.781524697793763
    - type: nauc_map_at_20_std
      value: 4.709613362008973
    - type: nauc_map_at_3_diff1
      value: 51.67730763922258
    - type: nauc_map_at_3_max
      value: 27.831209071832575
    - type: nauc_map_at_3_std
      value: 1.9547003342769491
    - type: nauc_map_at_5_diff1
      value: 50.94456118433537
    - type: nauc_map_at_5_max
      value: 27.62302486275545
    - type: nauc_map_at_5_std
      value: 2.709141109421837
    - type: nauc_mrr_at_1000_diff1
      value: 49.285265336265674
    - type: nauc_mrr_at_1000_max
      value: 28.313419759625337
    - type: nauc_mrr_at_1000_std
      value: 9.047308712658102
    - type: nauc_mrr_at_100_diff1
      value: 49.24923536605676
    - type: nauc_mrr_at_100_max
      value: 28.298612746077545
    - type: nauc_mrr_at_100_std
      value: 9.054497152985347
    - type: nauc_mrr_at_10_diff1
      value: 49.10691756769261
    - type: nauc_mrr_at_10_max
      value: 28.23586150111949
    - type: nauc_mrr_at_10_std
      value: 8.901423703472938
    - type: nauc_mrr_at_1_diff1
      value: 52.25138025277375
    - type: nauc_mrr_at_1_max
      value: 26.355056676288395
    - type: nauc_mrr_at_1_std
      value: 6.430376163405345
    - type: nauc_mrr_at_20_diff1
      value: 49.28169400715884
    - type: nauc_mrr_at_20_max
      value: 28.39434419567281
    - type: nauc_mrr_at_20_std
      value: 9.10118514648555
    - type: nauc_mrr_at_3_diff1
      value: 50.64479122666028
    - type: nauc_mrr_at_3_max
      value: 28.832312590631588
    - type: nauc_mrr_at_3_std
      value: 7.990475812682553
    - type: nauc_mrr_at_5_diff1
      value: 50.017578276500984
    - type: nauc_mrr_at_5_max
      value: 28.656357640735653
    - type: nauc_mrr_at_5_std
      value: 8.608087249565767
    - type: nauc_ndcg_at_1000_diff1
      value: 49.104936290947265
    - type: nauc_ndcg_at_1000_max
      value: 28.619121962227894
    - type: nauc_ndcg_at_1000_std
      value: 9.757408343308253
    - type: nauc_ndcg_at_100_diff1
      value: 48.13680462826767
    - type: nauc_ndcg_at_100_max
      value: 28.181225529870257
    - type: nauc_ndcg_at_100_std
      value: 10.001139771795788
    - type: nauc_ndcg_at_10_diff1
      value: 48.71650600628425
    - type: nauc_ndcg_at_10_max
      value: 27.62968625122465
    - type: nauc_ndcg_at_10_std
      value: 7.737164151185029
    - type: nauc_ndcg_at_1_diff1
      value: 52.25138025277375
    - type: nauc_ndcg_at_1_max
      value: 26.355056676288395
    - type: nauc_ndcg_at_1_std
      value: 6.430376163405345
    - type: nauc_ndcg_at_20_diff1
      value: 48.79300281701873
    - type: nauc_ndcg_at_20_max
      value: 28.385914629782107
    - type: nauc_ndcg_at_20_std
      value: 8.60238906023919
    - type: nauc_ndcg_at_3_diff1
      value: 51.4190264507031
    - type: nauc_ndcg_at_3_max
      value: 29.1265688820524
    - type: nauc_ndcg_at_3_std
      value: 6.578917291792241
    - type: nauc_ndcg_at_5_diff1
      value: 50.288543382090346
    - type: nauc_ndcg_at_5_max
      value: 28.182217394848408
    - type: nauc_ndcg_at_5_std
      value: 7.022145810060216
    - type: nauc_precision_at_1000_diff1
      value: -7.972044416677482
    - type: nauc_precision_at_1000_max
      value: -10.680209992690665
    - type: nauc_precision_at_1000_std
      value: 28.080291299709785
    - type: nauc_precision_at_100_diff1
      value: 3.9948049389828597
    - type: nauc_precision_at_100_max
      value: 1.7220637960609488
    - type: nauc_precision_at_100_std
      value: 35.971091376893824
    - type: nauc_precision_at_10_diff1
      value: 23.276306907078144
    - type: nauc_precision_at_10_max
      value: 18.20989779938144
    - type: nauc_precision_at_10_std
      value: 25.373674319451744
    - type: nauc_precision_at_1_diff1
      value: 52.25138025277375
    - type: nauc_precision_at_1_max
      value: 26.355056676288395
    - type: nauc_precision_at_1_std
      value: 6.430376163405345
    - type: nauc_precision_at_20_diff1
      value: 18.232942011308168
    - type: nauc_precision_at_20_max
      value: 16.16153444107366
    - type: nauc_precision_at_20_std
      value: 30.586025950631274
    - type: nauc_precision_at_3_diff1
      value: 37.63322651296769
    - type: nauc_precision_at_3_max
      value: 27.13718142601207
    - type: nauc_precision_at_3_std
      value: 13.647892479314633
    - type: nauc_precision_at_5_diff1
      value: 32.38454066880137
    - type: nauc_precision_at_5_max
      value: 24.559641952440664
    - type: nauc_precision_at_5_std
      value: 18.259272792921344
    - type: nauc_recall_at_1000_diff1
      value: 35.24269437060305
    - type: nauc_recall_at_1000_max
      value: 40.69594465175928
    - type: nauc_recall_at_1000_std
      value: 47.36300788402723
    - type: nauc_recall_at_100_diff1
      value: 29.83150262510703
    - type: nauc_recall_at_100_max
      value: 23.711760507130354
    - type: nauc_recall_at_100_std
      value: 24.7698700528816
    - type: nauc_recall_at_10_diff1
      value: 40.63474075281228
    - type: nauc_recall_at_10_max
      value: 26.912392247901966
    - type: nauc_recall_at_10_std
      value: 8.831088984195794
    - type: nauc_recall_at_1_diff1
      value: 54.85652943645487
    - type: nauc_recall_at_1_max
      value: 24.50498114129174
    - type: nauc_recall_at_1_std
      value: 0.3799552038716688
    - type: nauc_recall_at_20_diff1
      value: 40.07253803361886
    - type: nauc_recall_at_20_max
      value: 27.163744246134492
    - type: nauc_recall_at_20_std
      value: 13.00244843736101
    - type: nauc_recall_at_3_diff1
      value: 48.53732625901422
    - type: nauc_recall_at_3_max
      value: 30.12745866850734
    - type: nauc_recall_at_3_std
      value: 3.3512486117682956
    - type: nauc_recall_at_5_diff1
      value: 44.815926627861344
    - type: nauc_recall_at_5_max
      value: 28.203678371171453
    - type: nauc_recall_at_5_std
      value: 5.569126723668287
    - type: ndcg_at_1
      value: 33.794000000000004
    - type: ndcg_at_10
      value: 42.276
    - type: ndcg_at_100
      value: 48.815
    - type: ndcg_at_1000
      value: 51.101
    - type: ndcg_at_20
      value: 44.866
    - type: ndcg_at_3
      value: 38.397999999999996
    - type: ndcg_at_5
      value: 40.06
    - type: precision_at_1
      value: 33.794000000000004
    - type: precision_at_10
      value: 7.885000000000001
    - type: precision_at_100
      value: 1.617
    - type: precision_at_1000
      value: 0.244
    - type: precision_at_20
      value: 5.148
    - type: precision_at_3
      value: 17.984
    - type: precision_at_5
      value: 12.568999999999999
    - type: recall_at_1
      value: 27.913
    - type: recall_at_10
      value: 51.690999999999995
    - type: recall_at_100
      value: 80.88499999999999
    - type: recall_at_1000
      value: 95.429
    - type: recall_at_20
      value: 61.287000000000006
    - type: recall_at_3
      value: 39.959
    - type: recall_at_5
      value: 45.033
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackWordpressRetrieval (default)
      revision: 4ffe81d471b1924886b33c7567bfb200e9eec5c4
      split: test
      type: mteb/cqadupstack-wordpress
    metrics:
    - type: main_score
      value: 34.686
    - type: map_at_1
      value: 22.588
    - type: map_at_10
      value: 30.067
    - type: map_at_100
      value: 31.135
    - type: map_at_1000
      value: 31.234
    - type: map_at_20
      value: 30.682
    - type: map_at_3
      value: 27.362
    - type: map_at_5
      value: 28.93
    - type: mrr_at_1
      value: 24.58410351201479
    - type: mrr_at_10
      value: 31.907402517384032
    - type: mrr_at_100
      value: 32.89289790552948
    - type: mrr_at_1000
      value: 32.96321814246583
    - type: mrr_at_20
      value: 32.4946062027948
    - type: mrr_at_3
      value: 29.328404189772023
    - type: mrr_at_5
      value: 30.788662969808993
    - type: nauc_map_at_1000_diff1
      value: 39.07518133297138
    - type: nauc_map_at_1000_max
      value: 18.786382396290993
    - type: nauc_map_at_1000_std
      value: 1.018765660249156
    - type: nauc_map_at_100_diff1
      value: 39.09190989217314
    - type: nauc_map_at_100_max
      value: 18.73587092010555
    - type: nauc_map_at_100_std
      value: 0.9928097516282189
    - type: nauc_map_at_10_diff1
      value: 39.07873862859133
    - type: nauc_map_at_10_max
      value: 19.067215212969703
    - type: nauc_map_at_10_std
      value: 0.23553667993539434
    - type: nauc_map_at_1_diff1
      value: 44.59932826019673
    - type: nauc_map_at_1_max
      value: 18.84717818084193
    - type: nauc_map_at_1_std
      value: 0.19513776269542255
    - type: nauc_map_at_20_diff1
      value: 38.98216089259965
    - type: nauc_map_at_20_max
      value: 18.84519129889602
    - type: nauc_map_at_20_std
      value: 0.665926529265827
    - type: nauc_map_at_3_diff1
      value: 39.0562183507231
    - type: nauc_map_at_3_max
      value: 18.86926365406297
    - type: nauc_map_at_3_std
      value: 0.17133843139070712
    - type: nauc_map_at_5_diff1
      value: 38.51733742743825
    - type: nauc_map_at_5_max
      value: 18.00005743415062
    - type: nauc_map_at_5_std
      value: -0.605993384711123
    - type: nauc_mrr_at_1000_diff1
      value: 39.911253261997466
    - type: nauc_mrr_at_1000_max
      value: 18.844118480809254
    - type: nauc_mrr_at_1000_std
      value: 0.9658814738113289
    - type: nauc_mrr_at_100_diff1
      value: 39.90952538983193
    - type: nauc_mrr_at_100_max
      value: 18.835283493568788
    - type: nauc_mrr_at_100_std
      value: 0.9394971561045568
    - type: nauc_mrr_at_10_diff1
      value: 39.877766123305534
    - type: nauc_mrr_at_10_max
      value: 19.071731674273778
    - type: nauc_mrr_at_10_std
      value: 0.3342765792851528
    - type: nauc_mrr_at_1_diff1
      value: 46.123558536068856
    - type: nauc_mrr_at_1_max
      value: 18.79345449396603
    - type: nauc_mrr_at_1_std
      value: 1.0820507040897427
    - type: nauc_mrr_at_20_diff1
      value: 39.79555687568754
    - type: nauc_mrr_at_20_max
      value: 18.870647470918424
    - type: nauc_mrr_at_20_std
      value: 0.6498760586007435
    - type: nauc_mrr_at_3_diff1
      value: 40.36708901294981
    - type: nauc_mrr_at_3_max
      value: 18.853936883027377
    - type: nauc_mrr_at_3_std
      value: 0.50331853437221
    - type: nauc_mrr_at_5_diff1
      value: 39.73616849351637
    - type: nauc_mrr_at_5_max
      value: 18.066125044002366
    - type: nauc_mrr_at_5_std
      value: -0.13130879794539987
    - type: nauc_ndcg_at_1000_diff1
      value: 37.351293040381215
    - type: nauc_ndcg_at_1000_max
      value: 19.46889622691054
    - type: nauc_ndcg_at_1000_std
      value: 4.287770108015916
    - type: nauc_ndcg_at_100_diff1
      value: 37.54552699449341
    - type: nauc_ndcg_at_100_max
      value: 18.580891326097817
    - type: nauc_ndcg_at_100_std
      value: 3.8807359742657677
    - type: nauc_ndcg_at_10_diff1
      value: 37.52924061306312
    - type: nauc_ndcg_at_10_max
      value: 19.69924713913095
    - type: nauc_ndcg_at_10_std
      value: 0.5946724351267106
    - type: nauc_ndcg_at_1_diff1
      value: 46.123558536068856
    - type: nauc_ndcg_at_1_max
      value: 18.79345449396603
    - type: nauc_ndcg_at_1_std
      value: 1.0820507040897427
    - type: nauc_ndcg_at_20_diff1
      value: 37.02390511799261
    - type: nauc_ndcg_at_20_max
      value: 18.890186711279938
    - type: nauc_ndcg_at_20_std
      value: 2.1417527944124717
    - type: nauc_ndcg_at_3_diff1
      value: 37.60658372527831
    - type: nauc_ndcg_at_3_max
      value: 19.044629750320137
    - type: nauc_ndcg_at_3_std
      value: 0.8891505159485764
    - type: nauc_ndcg_at_5_diff1
      value: 36.52591325755527
    - type: nauc_ndcg_at_5_max
      value: 17.471901328312406
    - type: nauc_ndcg_at_5_std
      value: -0.8694111199060547
    - type: nauc_precision_at_1000_diff1
      value: -10.30516545212363
    - type: nauc_precision_at_1000_max
      value: -2.9659406379263733
    - type: nauc_precision_at_1000_std
      value: 10.644707220802445
    - type: nauc_precision_at_100_diff1
      value: 18.41020860453091
    - type: nauc_precision_at_100_max
      value: 11.116777949198559
    - type: nauc_precision_at_100_std
      value: 20.667411633330875
    - type: nauc_precision_at_10_diff1
      value: 31.519554677944793
    - type: nauc_precision_at_10_max
      value: 22.15649534599777
    - type: nauc_precision_at_10_std
      value: 7.979212837758018
    - type: nauc_precision_at_1_diff1
      value: 46.123558536068856
    - type: nauc_precision_at_1_max
      value: 18.79345449396603
    - type: nauc_precision_at_1_std
      value: 1.0820507040897427
    - type: nauc_precision_at_20_diff1
      value: 25.512505021115345
    - type: nauc_precision_at_20_max
      value: 17.677863977033507
    - type: nauc_precision_at_20_std
      value: 13.567270739738237
    - type: nauc_precision_at_3_diff1
      value: 33.761637394386554
    - type: nauc_precision_at_3_max
      value: 19.507943070005673
    - type: nauc_precision_at_3_std
      value: 4.16235441691752
    - type: nauc_precision_at_5_diff1
      value: 30.66266153952448
    - type: nauc_precision_at_5_max
      value: 16.220424873595434
    - type: nauc_precision_at_5_std
      value: 1.8392464410039773
    - type: nauc_recall_at_1000_diff1
      value: 19.06481725767347
    - type: nauc_recall_at_1000_max
      value: 28.67838805796927
    - type: nauc_recall_at_1000_std
      value: 41.32734487370867
    - type: nauc_recall_at_100_diff1
      value: 29.635010747820196
    - type: nauc_recall_at_100_max
      value: 16.04982794805439
    - type: nauc_recall_at_100_std
      value: 18.05594680109131
    - type: nauc_recall_at_10_diff1
      value: 32.004299928335676
    - type: nauc_recall_at_10_max
      value: 20.97469907327675
    - type: nauc_recall_at_10_std
      value: 1.2825598701093601
    - type: nauc_recall_at_1_diff1
      value: 44.59932826019673
    - type: nauc_recall_at_1_max
      value: 18.84717818084193
    - type: nauc_recall_at_1_std
      value: 0.19513776269542255
    - type: nauc_recall_at_20_diff1
      value: 29.23038587725173
    - type: nauc_recall_at_20_max
      value: 17.619541038813583
    - type: nauc_recall_at_20_std
      value: 7.292026542366455
    - type: nauc_recall_at_3_diff1
      value: 31.93598216934646
    - type: nauc_recall_at_3_max
      value: 18.664709946713902
    - type: nauc_recall_at_3_std
      value: 0.4038188098502496
    - type: nauc_recall_at_5_diff1
      value: 29.36987620910157
    - type: nauc_recall_at_5_max
      value: 15.352112069609989
    - type: nauc_recall_at_5_std
      value: -2.6567508595796254
    - type: ndcg_at_1
      value: 24.584
    - type: ndcg_at_10
      value: 34.686
    - type: ndcg_at_100
      value: 39.872
    - type: ndcg_at_1000
      value: 42.277
    - type: ndcg_at_20
      value: 36.882999999999996
    - type: ndcg_at_3
      value: 29.469
    - type: ndcg_at_5
      value: 32.108
    - type: precision_at_1
      value: 24.584
    - type: precision_at_10
      value: 5.453
    - type: precision_at_100
      value: 0.852
    - type: precision_at_1000
      value: 0.116
    - type: precision_at_20
      value: 3.253
    - type: precision_at_3
      value: 12.323
    - type: precision_at_5
      value: 8.909
    - type: recall_at_1
      value: 22.588
    - type: recall_at_10
      value: 47.160000000000004
    - type: recall_at_100
      value: 70.55
    - type: recall_at_1000
      value: 88.335
    - type: recall_at_20
      value: 55.476000000000006
    - type: recall_at_3
      value: 33.327
    - type: recall_at_5
      value: 39.772
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ClimateFEVER (default)
      revision: 47f2ac6acb640fc46020b02a5b59fdda04d39380
      split: test
      type: mteb/climate-fever
    metrics:
    - type: main_score
      value: 32.07
    - type: map_at_1
      value: 13.898
    - type: map_at_10
      value: 23.275000000000002
    - type: map_at_100
      value: 25.118000000000002
    - type: map_at_1000
      value: 25.288
    - type: map_at_20
      value: 24.305
    - type: map_at_3
      value: 19.358
    - type: map_at_5
      value: 21.477
    - type: mrr_at_1
      value: 31.335504885993487
    - type: mrr_at_10
      value: 42.7457990796753
    - type: mrr_at_100
      value: 43.60076325157638
    - type: mrr_at_1000
      value: 43.63709337856271
    - type: mrr_at_20
      value: 43.29067666012886
    - type: mrr_at_3
      value: 39.50054288816504
    - type: mrr_at_5
      value: 41.575461454940275
    - type: nauc_map_at_1000_diff1
      value: 23.601282518550242
    - type: nauc_map_at_1000_max
      value: 31.82806145593328
    - type: nauc_map_at_1000_std
      value: 13.729377093521592
    - type: nauc_map_at_100_diff1
      value: 23.582444310674465
    - type: nauc_map_at_100_max
      value: 31.804656616265188
    - type: nauc_map_at_100_std
      value: 13.668148635676522
    - type: nauc_map_at_10_diff1
      value: 23.949699958740194
    - type: nauc_map_at_10_max
      value: 31.726072454793712
    - type: nauc_map_at_10_std
      value: 12.78334014344823
    - type: nauc_map_at_1_diff1
      value: 30.616245340729424
    - type: nauc_map_at_1_max
      value: 29.643210336657326
    - type: nauc_map_at_1_std
      value: 10.982673808754015
    - type: nauc_map_at_20_diff1
      value: 23.722777638167738
    - type: nauc_map_at_20_max
      value: 31.903283136606014
    - type: nauc_map_at_20_std
      value: 13.32131655673745
    - type: nauc_map_at_3_diff1
      value: 25.36053723028792
    - type: nauc_map_at_3_max
      value: 30.737610916133402
    - type: nauc_map_at_3_std
      value: 11.78505026942106
    - type: nauc_map_at_5_diff1
      value: 24.470950874444412
    - type: nauc_map_at_5_max
      value: 31.0798656022895
    - type: nauc_map_at_5_std
      value: 11.960293083591864
    - type: nauc_mrr_at_1000_diff1
      value: 22.519657500882627
    - type: nauc_mrr_at_1000_max
      value: 27.650895941041355
    - type: nauc_mrr_at_1000_std
      value: 12.365888045660855
    - type: nauc_mrr_at_100_diff1
      value: 22.51541726496946
    - type: nauc_mrr_at_100_max
      value: 27.653515027519187
    - type: nauc_mrr_at_100_std
      value: 12.366433985624353
    - type: nauc_mrr_at_10_diff1
      value: 22.49453492661782
    - type: nauc_mrr_at_10_max
      value: 27.687781708425458
    - type: nauc_mrr_at_10_std
      value: 12.128487278748468
    - type: nauc_mrr_at_1_diff1
      value: 25.92491417010959
    - type: nauc_mrr_at_1_max
      value: 27.48929080837516
    - type: nauc_mrr_at_1_std
      value: 11.931616279055087
    - type: nauc_mrr_at_20_diff1
      value: 22.390218406658654
    - type: nauc_mrr_at_20_max
      value: 27.605818788760146
    - type: nauc_mrr_at_20_std
      value: 12.312373444698741
    - type: nauc_mrr_at_3_diff1
      value: 22.383138582317557
    - type: nauc_mrr_at_3_max
      value: 26.99323446984345
    - type: nauc_mrr_at_3_std
      value: 11.687809223109353
    - type: nauc_mrr_at_5_diff1
      value: 22.397896472667494
    - type: nauc_mrr_at_5_max
      value: 27.63083970846884
    - type: nauc_mrr_at_5_std
      value: 12.160529102640488
    - type: nauc_ndcg_at_1000_diff1
      value: 21.31738163655189
    - type: nauc_ndcg_at_1000_max
      value: 31.301211411542617
    - type: nauc_ndcg_at_1000_std
      value: 16.720668965584487
    - type: nauc_ndcg_at_100_diff1
      value: 21.118866773597762
    - type: nauc_ndcg_at_100_max
      value: 31.085373015863972
    - type: nauc_ndcg_at_100_std
      value: 16.138003678278878
    - type: nauc_ndcg_at_10_diff1
      value: 22.026498818115638
    - type: nauc_ndcg_at_10_max
      value: 31.506997290510935
    - type: nauc_ndcg_at_10_std
      value: 13.57096592159118
    - type: nauc_ndcg_at_1_diff1
      value: 25.92491417010959
    - type: nauc_ndcg_at_1_max
      value: 27.48929080837516
    - type: nauc_ndcg_at_1_std
      value: 11.931616279055087
    - type: nauc_ndcg_at_20_diff1
      value: 21.443454032261748
    - type: nauc_ndcg_at_20_max
      value: 31.609366427889686
    - type: nauc_ndcg_at_20_std
      value: 14.736618427155305
    - type: nauc_ndcg_at_3_diff1
      value: 22.99396762461139
    - type: nauc_ndcg_at_3_max
      value: 28.938428425331498
    - type: nauc_ndcg_at_3_std
      value: 11.664760840706652
    - type: nauc_ndcg_at_5_diff1
      value: 22.552441872215653
    - type: nauc_ndcg_at_5_max
      value: 30.452760185884337
    - type: nauc_ndcg_at_5_std
      value: 12.360254050201295
    - type: nauc_precision_at_1000_diff1
      value: -3.9683924059407314
    - type: nauc_precision_at_1000_max
      value: 3.1135339617664943
    - type: nauc_precision_at_1000_std
      value: 14.049522748730524
    - type: nauc_precision_at_100_diff1
      value: 1.214266654713573
    - type: nauc_precision_at_100_max
      value: 11.063713013371185
    - type: nauc_precision_at_100_std
      value: 15.46820195946926
    - type: nauc_precision_at_10_diff1
      value: 8.77954221711442
    - type: nauc_precision_at_10_max
      value: 23.39921068596534
    - type: nauc_precision_at_10_std
      value: 13.587649615468699
    - type: nauc_precision_at_1_diff1
      value: 25.92491417010959
    - type: nauc_precision_at_1_max
      value: 27.48929080837516
    - type: nauc_precision_at_1_std
      value: 11.931616279055087
    - type: nauc_precision_at_20_diff1
      value: 6.036967227349606
    - type: nauc_precision_at_20_max
      value: 20.171763639015346
    - type: nauc_precision_at_20_std
      value: 15.17095139145748
    - type: nauc_precision_at_3_diff1
      value: 14.315496810753443
    - type: nauc_precision_at_3_max
      value: 26.03925092361848
    - type: nauc_precision_at_3_std
      value: 12.895260164293143
    - type: nauc_precision_at_5_diff1
      value: 11.952135077045405
    - type: nauc_precision_at_5_max
      value: 25.321138362510364
    - type: nauc_precision_at_5_std
      value: 12.414454072333383
    - type: nauc_recall_at_1000_diff1
      value: 8.25128781606991
    - type: nauc_recall_at_1000_max
      value: 24.885123319470097
    - type: nauc_recall_at_1000_std
      value: 25.66656170122141
    - type: nauc_recall_at_100_diff1
      value: 10.736057820655297
    - type: nauc_recall_at_100_max
      value: 24.96180736059367
    - type: nauc_recall_at_100_std
      value: 20.20503176227293
    - type: nauc_recall_at_10_diff1
      value: 16.190581044786654
    - type: nauc_recall_at_10_max
      value: 29.7832682887181
    - type: nauc_recall_at_10_std
      value: 13.418776085336537
    - type: nauc_recall_at_1_diff1
      value: 30.616245340729424
    - type: nauc_recall_at_1_max
      value: 29.643210336657326
    - type: nauc_recall_at_1_std
      value: 10.982673808754015
    - type: nauc_recall_at_20_diff1
      value: 13.801030136320158
    - type: nauc_recall_at_20_max
      value: 28.696012038008778
    - type: nauc_recall_at_20_std
      value: 15.63498566995645
    - type: nauc_recall_at_3_diff1
      value: 20.77726592072591
    - type: nauc_recall_at_3_max
      value: 28.60982315119743
    - type: nauc_recall_at_3_std
      value: 11.007238770363177
    - type: nauc_recall_at_5_diff1
      value: 18.26388072543542
    - type: nauc_recall_at_5_max
      value: 28.853182826840634
    - type: nauc_recall_at_5_std
      value: 11.529192291272095
    - type: ndcg_at_1
      value: 31.336000000000002
    - type: ndcg_at_10
      value: 32.07
    - type: ndcg_at_100
      value: 39.018
    - type: ndcg_at_1000
      value: 42.065000000000005
    - type: ndcg_at_20
      value: 34.934
    - type: ndcg_at_3
      value: 26.362999999999996
    - type: ndcg_at_5
      value: 28.557
    - type: precision_at_1
      value: 31.336000000000002
    - type: precision_at_10
      value: 9.876
    - type: precision_at_100
      value: 1.7309999999999999
    - type: precision_at_1000
      value: 0.22999999999999998
    - type: precision_at_20
      value: 6.1530000000000005
    - type: precision_at_3
      value: 19.37
    - type: precision_at_5
      value: 15.062000000000001
    - type: recall_at_1
      value: 13.898
    - type: recall_at_10
      value: 37.849
    - type: recall_at_100
      value: 61.532
    - type: recall_at_1000
      value: 78.533
    - type: recall_at_20
      value: 46.013999999999996
    - type: recall_at_3
      value: 23.719
    - type: recall_at_5
      value: 30.078
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB DBPedia (default)
      revision: c0f706b76e590d620bd6618b3ca8efdd34e2d659
      split: test
      type: mteb/dbpedia
    metrics:
    - type: main_score
      value: 41.509
    - type: map_at_1
      value: 8.971
    - type: map_at_10
      value: 20.008
    - type: map_at_100
      value: 28.46
    - type: map_at_1000
      value: 29.979
    - type: map_at_20
      value: 23.408
    - type: map_at_3
      value: 14.216000000000001
    - type: map_at_5
      value: 16.441
    - type: mrr_at_1
      value: 68.75
    - type: mrr_at_10
      value: 76.0234126984127
    - type: mrr_at_100
      value: 76.37391158142093
    - type: mrr_at_1000
      value: 76.38523336924527
    - type: mrr_at_20
      value: 76.2691696933957
    - type: mrr_at_3
      value: 74.45833333333333
    - type: mrr_at_5
      value: 75.43333333333334
    - type: nauc_map_at_1000_diff1
      value: 20.71635270857788
    - type: nauc_map_at_1000_max
      value: 25.42869728067827
    - type: nauc_map_at_1000_std
      value: 21.100175366479522
    - type: nauc_map_at_100_diff1
      value: 20.608494763870222
    - type: nauc_map_at_100_max
      value: 22.833986238143908
    - type: nauc_map_at_100_std
      value: 17.62715884938892
    - type: nauc_map_at_10_diff1
      value: 21.447759662578054
    - type: nauc_map_at_10_max
      value: 7.167359276352099
    - type: nauc_map_at_10_std
      value: -9.160037778272423
    - type: nauc_map_at_1_diff1
      value: 30.24216030425471
    - type: nauc_map_at_1_max
      value: -4.081674590768062
    - type: nauc_map_at_1_std
      value: -26.59043516841042
    - type: nauc_map_at_20_diff1
      value: 21.656522421844592
    - type: nauc_map_at_20_max
      value: 13.681271369307652
    - type: nauc_map_at_20_std
      value: 0.17494449190714542
    - type: nauc_map_at_3_diff1
      value: 23.979978408164452
    - type: nauc_map_at_3_max
      value: -0.5771962120180582
    - type: nauc_map_at_3_std
      value: -21.039815038431716
    - type: nauc_map_at_5_diff1
      value: 22.289707568753904
    - type: nauc_map_at_5_max
      value: 2.15399887276463
    - type: nauc_map_at_5_std
      value: -16.303144563200952
    - type: nauc_mrr_at_1000_diff1
      value: 49.766235515625404
    - type: nauc_mrr_at_1000_max
      value: 62.80559880239637
    - type: nauc_mrr_at_1000_std
      value: 41.17500922109195
    - type: nauc_mrr_at_100_diff1
      value: 49.74559867250971
    - type: nauc_mrr_at_100_max
      value: 62.812094391808934
    - type: nauc_mrr_at_100_std
      value: 41.19518622562108
    - type: nauc_mrr_at_10_diff1
      value: 49.72978828074041
    - type: nauc_mrr_at_10_max
      value: 63.00656167695157
    - type: nauc_mrr_at_10_std
      value: 41.32518900864204
    - type: nauc_mrr_at_1_diff1
      value: 53.72387323171398
    - type: nauc_mrr_at_1_max
      value: 62.177431735935926
    - type: nauc_mrr_at_1_std
      value: 39.0257703695581
    - type: nauc_mrr_at_20_diff1
      value: 49.781691608651094
    - type: nauc_mrr_at_20_max
      value: 62.84092904839108
    - type: nauc_mrr_at_20_std
      value: 41.14893733570565
    - type: nauc_mrr_at_3_diff1
      value: 49.3798438227573
    - type: nauc_mrr_at_3_max
      value: 61.36198512178078
    - type: nauc_mrr_at_3_std
      value: 39.77785476155713
    - type: nauc_mrr_at_5_diff1
      value: 49.36345122563626
    - type: nauc_mrr_at_5_max
      value: 62.45251850110982
    - type: nauc_mrr_at_5_std
      value: 41.43332050821922
    - type: nauc_ndcg_at_1000_diff1
      value: 24.696693686275996
    - type: nauc_ndcg_at_1000_max
      value: 38.44309797486417
    - type: nauc_ndcg_at_1000_std
      value: 35.2874797010304
    - type: nauc_ndcg_at_100_diff1
      value: 24.168419610949236
    - type: nauc_ndcg_at_100_max
      value: 31.860261024845034
    - type: nauc_ndcg_at_100_std
      value: 27.071469994294638
    - type: nauc_ndcg_at_10_diff1
      value: 25.44647741145385
    - type: nauc_ndcg_at_10_max
      value: 34.076193050453334
    - type: nauc_ndcg_at_10_std
      value: 22.663252359161266
    - type: nauc_ndcg_at_1_diff1
      value: 48.643715402331395
    - type: nauc_ndcg_at_1_max
      value: 50.70791075807851
    - type: nauc_ndcg_at_1_std
      value: 26.94596138125655
    - type: nauc_ndcg_at_20_diff1
      value: 26.121466047462455
    - type: nauc_ndcg_at_20_max
      value: 31.421716794979275
    - type: nauc_ndcg_at_20_std
      value: 19.275517839979756
    - type: nauc_ndcg_at_3_diff1
      value: 32.25895577491697
    - type: nauc_ndcg_at_3_max
      value: 41.430079316151776
    - type: nauc_ndcg_at_3_std
      value: 27.723858260770484
    - type: nauc_ndcg_at_5_diff1
      value: 27.009852393094665
    - type: nauc_ndcg_at_5_max
      value: 38.258172708921215
    - type: nauc_ndcg_at_5_std
      value: 27.718345574181225
    - type: nauc_precision_at_1000_diff1
      value: -5.4475159651809415
    - type: nauc_precision_at_1000_max
      value: 33.00979585907006
    - type: nauc_precision_at_1000_std
      value: 46.07431006127377
    - type: nauc_precision_at_100_diff1
      value: 0.3103675855576706
    - type: nauc_precision_at_100_max
      value: 41.92524296731323
    - type: nauc_precision_at_100_std
      value: 60.757528138851626
    - type: nauc_precision_at_10_diff1
      value: 5.355902759114419
    - type: nauc_precision_at_10_max
      value: 43.13634536123484
    - type: nauc_precision_at_10_std
      value: 51.716679602591185
    - type: nauc_precision_at_1_diff1
      value: 53.72387323171398
    - type: nauc_precision_at_1_max
      value: 62.177431735935926
    - type: nauc_precision_at_1_std
      value: 39.0257703695581
    - type: nauc_precision_at_20_diff1
      value: 5.372617435599063
    - type: nauc_precision_at_20_max
      value: 44.98279636447302
    - type: nauc_precision_at_20_std
      value: 56.44722227917176
    - type: nauc_precision_at_3_diff1
      value: 17.726040053724347
    - type: nauc_precision_at_3_max
      value: 46.460723339578394
    - type: nauc_precision_at_3_std
      value: 43.34419797022352
    - type: nauc_precision_at_5_diff1
      value: 8.45438036902179
    - type: nauc_precision_at_5_max
      value: 42.3923372945306
    - type: nauc_precision_at_5_std
      value: 47.49390765988281
    - type: nauc_recall_at_1000_diff1
      value: 11.917974004221767
    - type: nauc_recall_at_1000_max
      value: 25.81822938313284
    - type: nauc_recall_at_1000_std
      value: 35.98947609543732
    - type: nauc_recall_at_100_diff1
      value: 12.911877503356678
    - type: nauc_recall_at_100_max
      value: 18.93599808823274
    - type: nauc_recall_at_100_std
      value: 21.731139853742217
    - type: nauc_recall_at_10_diff1
      value: 13.904367292829209
    - type: nauc_recall_at_10_max
      value: 1.3333050638442192
    - type: nauc_recall_at_10_std
      value: -14.821511640639176
    - type: nauc_recall_at_1_diff1
      value: 30.24216030425471
    - type: nauc_recall_at_1_max
      value: -4.081674590768062
    - type: nauc_recall_at_1_std
      value: -26.59043516841042
    - type: nauc_recall_at_20_diff1
      value: 14.91771956958437
    - type: nauc_recall_at_20_max
      value: 6.7473141995412895
    - type: nauc_recall_at_20_std
      value: -6.265357731137971
    - type: nauc_recall_at_3_diff1
      value: 18.66993022017212
    - type: nauc_recall_at_3_max
      value: -4.357039857583967
    - type: nauc_recall_at_3_std
      value: -23.74889016078873
    - type: nauc_recall_at_5_diff1
      value: 16.472727249477852
    - type: nauc_recall_at_5_max
      value: -2.0420602527912046
    - type: nauc_recall_at_5_std
      value: -19.746298392018343
    - type: ndcg_at_1
      value: 55.75
    - type: ndcg_at_10
      value: 41.509
    - type: ndcg_at_100
      value: 46.33
    - type: ndcg_at_1000
      value: 53.59
    - type: ndcg_at_20
      value: 41.327999999999996
    - type: ndcg_at_3
      value: 46.465
    - type: ndcg_at_5
      value: 43.35
    - type: precision_at_1
      value: 68.75
    - type: precision_at_10
      value: 33.2
    - type: precision_at_100
      value: 10.325
    - type: precision_at_1000
      value: 1.939
    - type: precision_at_20
      value: 25.324999999999996
    - type: precision_at_3
      value: 50.5
    - type: precision_at_5
      value: 42.3
    - type: recall_at_1
      value: 8.971
    - type: recall_at_10
      value: 26.046000000000003
    - type: recall_at_100
      value: 53.08500000000001
    - type: recall_at_1000
      value: 76.385
    - type: recall_at_20
      value: 34.103
    - type: recall_at_3
      value: 15.586
    - type: recall_at_5
      value: 18.913
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB EmotionClassification (default)
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
      split: test
      type: mteb/emotion
    metrics:
    - type: accuracy
      value: 89.185
    - type: f1
      value: 85.28981308260418
    - type: f1_weighted
      value: 89.69483129559028
    - type: main_score
      value: 89.185
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB FEVER (default)
      revision: bea83ef9e8fb933d90a2f1d5515737465d613e12
      split: test
      type: mteb/fever
    metrics:
    - type: main_score
      value: 89.468
    - type: map_at_1
      value: 78.12
    - type: map_at_10
      value: 86.10600000000001
    - type: map_at_100
      value: 86.287
    - type: map_at_1000
      value: 86.299
    - type: map_at_20
      value: 86.22800000000001
    - type: map_at_3
      value: 85.221
    - type: map_at_5
      value: 85.788
    - type: mrr_at_1
      value: 84.32343234323433
    - type: mrr_at_10
      value: 90.7537122759895
    - type: mrr_at_100
      value: 90.79053643472608
    - type: mrr_at_1000
      value: 90.79091280996353
    - type: mrr_at_20
      value: 90.78692883197269
    - type: mrr_at_3
      value: 90.2940294029403
    - type: mrr_at_5
      value: 90.65256525652565
    - type: nauc_map_at_1000_diff1
      value: 47.814882427963
    - type: nauc_map_at_1000_max
      value: 16.967407795372537
    - type: nauc_map_at_1000_std
      value: -11.483422834029062
    - type: nauc_map_at_100_diff1
      value: 47.76939643162608
    - type: nauc_map_at_100_max
      value: 16.951025999310545
    - type: nauc_map_at_100_std
      value: -11.489808864940063
    - type: nauc_map_at_10_diff1
      value: 47.30413407688185
    - type: nauc_map_at_10_max
      value: 16.613115501282856
    - type: nauc_map_at_10_std
      value: -11.759246765180155
    - type: nauc_map_at_1_diff1
      value: 53.6006484769898
    - type: nauc_map_at_1_max
      value: 14.682792641946948
    - type: nauc_map_at_1_std
      value: -12.000758799884249
    - type: nauc_map_at_20_diff1
      value: 47.524968835385764
    - type: nauc_map_at_20_max
      value: 16.729916511120678
    - type: nauc_map_at_20_std
      value: -11.653043621830538
    - type: nauc_map_at_3_diff1
      value: 46.750049161763044
    - type: nauc_map_at_3_max
      value: 16.432714899161304
    - type: nauc_map_at_3_std
      value: -12.653734698170267
    - type: nauc_map_at_5_diff1
      value: 46.85187589976551
    - type: nauc_map_at_5_max
      value: 16.239837127455655
    - type: nauc_map_at_5_std
      value: -12.341125124855436
    - type: nauc_mrr_at_1000_diff1
      value: 70.85509728651267
    - type: nauc_mrr_at_1000_max
      value: 24.942434061539124
    - type: nauc_mrr_at_1000_std
      value: -16.21201556600899
    - type: nauc_mrr_at_100_diff1
      value: 70.85473292653802
    - type: nauc_mrr_at_100_max
      value: 24.94460650144953
    - type: nauc_mrr_at_100_std
      value: -16.207645432220186
    - type: nauc_mrr_at_10_diff1
      value: 70.84343487511488
    - type: nauc_mrr_at_10_max
      value: 25.056290885599118
    - type: nauc_mrr_at_10_std
      value: -16.309304644657942
    - type: nauc_mrr_at_1_diff1
      value: 72.05772737088373
    - type: nauc_mrr_at_1_max
      value: 23.333429518350318
    - type: nauc_mrr_at_1_std
      value: -13.922526573612567
    - type: nauc_mrr_at_20_diff1
      value: 70.85190227487337
    - type: nauc_mrr_at_20_max
      value: 24.946003793124024
    - type: nauc_mrr_at_20_std
      value: -16.227179752764812
    - type: nauc_mrr_at_3_diff1
      value: 70.39386014931581
    - type: nauc_mrr_at_3_max
      value: 25.177891210652863
    - type: nauc_mrr_at_3_std
      value: -17.1940275674501
    - type: nauc_mrr_at_5_diff1
      value: 70.63168332951341
    - type: nauc_mrr_at_5_max
      value: 25.152966502751152
    - type: nauc_mrr_at_5_std
      value: -16.634033834694694
    - type: nauc_ndcg_at_1000_diff1
      value: 50.41741948292622
    - type: nauc_ndcg_at_1000_max
      value: 19.11215121414335
    - type: nauc_ndcg_at_1000_std
      value: -10.407613005057842
    - type: nauc_ndcg_at_100_diff1
      value: 49.3915447919262
    - type: nauc_ndcg_at_100_max
      value: 18.83822812672437
    - type: nauc_ndcg_at_100_std
      value: -10.313117416448005
    - type: nauc_ndcg_at_10_diff1
      value: 47.482404042995
    - type: nauc_ndcg_at_10_max
      value: 17.581992141280356
    - type: nauc_ndcg_at_10_std
      value: -11.652825111338238
    - type: nauc_ndcg_at_1_diff1
      value: 72.05772737088373
    - type: nauc_ndcg_at_1_max
      value: 23.333429518350318
    - type: nauc_ndcg_at_1_std
      value: -13.922526573612567
    - type: nauc_ndcg_at_20_diff1
      value: 48.058080306478615
    - type: nauc_ndcg_at_20_max
      value: 17.73032152398436
    - type: nauc_ndcg_at_20_std
      value: -11.245366189050753
    - type: nauc_ndcg_at_3_diff1
      value: 48.171657249342694
    - type: nauc_ndcg_at_3_max
      value: 18.225809988920734
    - type: nauc_ndcg_at_3_std
      value: -13.726690658155313
    - type: nauc_ndcg_at_5_diff1
      value: 47.0205970012977
    - type: nauc_ndcg_at_5_max
      value: 17.207332049361316
    - type: nauc_ndcg_at_5_std
      value: -12.974784558692063
    - type: nauc_precision_at_1000_diff1
      value: 1.1284800699700706
    - type: nauc_precision_at_1000_max
      value: 11.464041305924374
    - type: nauc_precision_at_1000_std
      value: 11.280161942399515
    - type: nauc_precision_at_100_diff1
      value: -0.4374443886093248
    - type: nauc_precision_at_100_max
      value: 13.532808305192246
    - type: nauc_precision_at_100_std
      value: 12.889347072854918
    - type: nauc_precision_at_10_diff1
      value: -3.311574121587653
    - type: nauc_precision_at_10_max
      value: 11.958917196342785
    - type: nauc_precision_at_10_std
      value: 6.761102693695978
    - type: nauc_precision_at_1_diff1
      value: 72.05772737088373
    - type: nauc_precision_at_1_max
      value: 23.333429518350318
    - type: nauc_precision_at_1_std
      value: -13.922526573612567
    - type: nauc_precision_at_20_diff1
      value: -4.213966257917642
    - type: nauc_precision_at_20_max
      value: 10.730863570189548
    - type: nauc_precision_at_20_std
      value: 8.320393111232448
    - type: nauc_precision_at_3_diff1
      value: 11.44044739528053
    - type: nauc_precision_at_3_max
      value: 19.88406160025192
    - type: nauc_precision_at_3_std
      value: -5.401918624214976
    - type: nauc_precision_at_5_diff1
      value: 0.17668330364694737
    - type: nauc_precision_at_5_max
      value: 13.674159890626383
    - type: nauc_precision_at_5_std
      value: -0.17207208349965347
    - type: nauc_recall_at_1000_diff1
      value: 4.431520335668279
    - type: nauc_recall_at_1000_max
      value: 23.05674849978065
    - type: nauc_recall_at_1000_std
      value: 38.1839303202116
    - type: nauc_recall_at_100_diff1
      value: 3.660595749363534
    - type: nauc_recall_at_100_max
      value: 17.1220755987757
    - type: nauc_recall_at_100_std
      value: 21.167581519286625
    - type: nauc_recall_at_10_diff1
      value: 7.941100563355342
    - type: nauc_recall_at_10_max
      value: 9.23944221490271
    - type: nauc_recall_at_10_std
      value: -2.0154754017498995
    - type: nauc_recall_at_1_diff1
      value: 53.6006484769898
    - type: nauc_recall_at_1_max
      value: 14.682792641946948
    - type: nauc_recall_at_1_std
      value: -12.000758799884249
    - type: nauc_recall_at_20_diff1
      value: 2.144737673969165
    - type: nauc_recall_at_20_max
      value: 7.515807202223332
    - type: nauc_recall_at_20_std
      value: 4.280373925464902
    - type: nauc_recall_at_3_diff1
      value: 21.99638259064024
    - type: nauc_recall_at_3_max
      value: 11.926648836638517
    - type: nauc_recall_at_3_std
      value: -13.135168470597506
    - type: nauc_recall_at_5_diff1
      value: 12.792900902829398
    - type: nauc_recall_at_5_max
      value: 8.635147490569679
    - type: nauc_recall_at_5_std
      value: -9.80199646054509
    - type: ndcg_at_1
      value: 84.32300000000001
    - type: ndcg_at_10
      value: 89.468
    - type: ndcg_at_100
      value: 90.026
    - type: ndcg_at_1000
      value: 90.226
    - type: ndcg_at_20
      value: 89.77199999999999
    - type: ndcg_at_3
      value: 88.24
    - type: ndcg_at_5
      value: 88.97
    - type: precision_at_1
      value: 84.32300000000001
    - type: precision_at_10
      value: 10.59
    - type: precision_at_100
      value: 1.1079999999999999
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_20
      value: 5.396999999999999
    - type: precision_at_3
      value: 33.533
    - type: precision_at_5
      value: 20.66
    - type: recall_at_1
      value: 78.12
    - type: recall_at_10
      value: 95.17999999999999
    - type: recall_at_100
      value: 97.195
    - type: recall_at_1000
      value: 98.404
    - type: recall_at_20
      value: 96.17
    - type: recall_at_3
      value: 91.843
    - type: recall_at_5
      value: 93.783
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB FiQA2018 (default)
      revision: 27a168819829fe9bcd655c2df245fb19452e8e06
      split: test
      type: mteb/fiqa
    metrics:
    - type: main_score
      value: 53.539
    - type: map_at_1
      value: 27.778999999999996
    - type: map_at_10
      value: 45.245000000000005
    - type: map_at_100
      value: 47.32
    - type: map_at_1000
      value: 47.454
    - type: map_at_20
      value: 46.467999999999996
    - type: map_at_3
      value: 39.757999999999996
    - type: map_at_5
      value: 42.809000000000005
    - type: mrr_at_1
      value: 54.32098765432099
    - type: mrr_at_10
      value: 62.30979325886733
    - type: mrr_at_100
      value: 62.92244797505688
    - type: mrr_at_1000
      value: 62.94596057377849
    - type: mrr_at_20
      value: 62.70844962533846
    - type: mrr_at_3
      value: 60.15946502057613
    - type: mrr_at_5
      value: 61.46347736625515
    - type: nauc_map_at_1000_diff1
      value: 45.50380289838926
    - type: nauc_map_at_1000_max
      value: 39.940645362903304
    - type: nauc_map_at_1000_std
      value: -10.811833950264708
    - type: nauc_map_at_100_diff1
      value: 45.442234064344795
    - type: nauc_map_at_100_max
      value: 39.881285749056225
    - type: nauc_map_at_100_std
      value: -10.823096394571396
    - type: nauc_map_at_10_diff1
      value: 45.33418238576907
    - type: nauc_map_at_10_max
      value: 38.24097832799157
    - type: nauc_map_at_10_std
      value: -12.519727514204664
    - type: nauc_map_at_1_diff1
      value: 46.62899301926139
    - type: nauc_map_at_1_max
      value: 21.89247441846697
    - type: nauc_map_at_1_std
      value: -14.202185813260924
    - type: nauc_map_at_20_diff1
      value: 45.323487098569856
    - type: nauc_map_at_20_max
      value: 39.27751392873174
    - type: nauc_map_at_20_std
      value: -11.322402070003502
    - type: nauc_map_at_3_diff1
      value: 45.979596859550284
    - type: nauc_map_at_3_max
      value: 32.802117882237944
    - type: nauc_map_at_3_std
      value: -13.710733808514739
    - type: nauc_map_at_5_diff1
      value: 44.80819713502716
    - type: nauc_map_at_5_max
      value: 35.740064539896615
    - type: nauc_map_at_5_std
      value: -12.902638507037972
    - type: nauc_mrr_at_1000_diff1
      value: 56.88392611765304
    - type: nauc_mrr_at_1000_max
      value: 50.47442897585142
    - type: nauc_mrr_at_1000_std
      value: -9.047188173877071
    - type: nauc_mrr_at_100_diff1
      value: 56.87463746927974
    - type: nauc_mrr_at_100_max
      value: 50.490307200732246
    - type: nauc_mrr_at_100_std
      value: -9.01348621430336
    - type: nauc_mrr_at_10_diff1
      value: 56.78417083488062
    - type: nauc_mrr_at_10_max
      value: 50.52437714740074
    - type: nauc_mrr_at_10_std
      value: -9.223013780870014
    - type: nauc_mrr_at_1_diff1
      value: 60.25493474957012
    - type: nauc_mrr_at_1_max
      value: 49.19917752206111
    - type: nauc_mrr_at_1_std
      value: -10.279865131882886
    - type: nauc_mrr_at_20_diff1
      value: 56.81725261916786
    - type: nauc_mrr_at_20_max
      value: 50.49830648058292
    - type: nauc_mrr_at_20_std
      value: -9.09743420683843
    - type: nauc_mrr_at_3_diff1
      value: 57.333278318964034
    - type: nauc_mrr_at_3_max
      value: 50.25705047802745
    - type: nauc_mrr_at_3_std
      value: -9.916642227286419
    - type: nauc_mrr_at_5_diff1
      value: 56.82073637830945
    - type: nauc_mrr_at_5_max
      value: 50.27830930234889
    - type: nauc_mrr_at_5_std
      value: -9.671091352500893
    - type: nauc_ndcg_at_1000_diff1
      value: 47.95819905489409
    - type: nauc_ndcg_at_1000_max
      value: 45.753161440198326
    - type: nauc_ndcg_at_1000_std
      value: -6.583018057539338
    - type: nauc_ndcg_at_100_diff1
      value: 47.291199318106045
    - type: nauc_ndcg_at_100_max
      value: 45.43586391412464
    - type: nauc_ndcg_at_100_std
      value: -5.99288348563384
    - type: nauc_ndcg_at_10_diff1
      value: 46.77840336870649
    - type: nauc_ndcg_at_10_max
      value: 42.340611222995506
    - type: nauc_ndcg_at_10_std
      value: -10.48527206785535
    - type: nauc_ndcg_at_1_diff1
      value: 60.25493474957012
    - type: nauc_ndcg_at_1_max
      value: 49.19917752206111
    - type: nauc_ndcg_at_1_std
      value: -10.279865131882886
    - type: nauc_ndcg_at_20_diff1
      value: 46.78177712186085
    - type: nauc_ndcg_at_20_max
      value: 43.83928471063127
    - type: nauc_ndcg_at_20_std
      value: -8.173306506160088
    - type: nauc_ndcg_at_3_diff1
      value: 48.36715093618337
    - type: nauc_ndcg_at_3_max
      value: 43.67796062466437
    - type: nauc_ndcg_at_3_std
      value: -12.045382742803556
    - type: nauc_ndcg_at_5_diff1
      value: 46.40051818654287
    - type: nauc_ndcg_at_5_max
      value: 41.36487853022696
    - type: nauc_ndcg_at_5_std
      value: -11.693829932059113
    - type: nauc_precision_at_1000_diff1
      value: 0.7796900642554564
    - type: nauc_precision_at_1000_max
      value: 28.477371585679222
    - type: nauc_precision_at_1000_std
      value: 11.898676086867058
    - type: nauc_precision_at_100_diff1
      value: 5.032984088999429
    - type: nauc_precision_at_100_max
      value: 35.33927789355232
    - type: nauc_precision_at_100_std
      value: 13.597744636874568
    - type: nauc_precision_at_10_diff1
      value: 18.229749237049223
    - type: nauc_precision_at_10_max
      value: 44.60575691827839
    - type: nauc_precision_at_10_std
      value: 2.132683900825309
    - type: nauc_precision_at_1_diff1
      value: 60.25493474957012
    - type: nauc_precision_at_1_max
      value: 49.19917752206111
    - type: nauc_precision_at_1_std
      value: -10.279865131882886
    - type: nauc_precision_at_20_diff1
      value: 12.432187941761182
    - type: nauc_precision_at_20_max
      value: 42.35945630665768
    - type: nauc_precision_at_20_std
      value: 8.328280384073741
    - type: nauc_precision_at_3_diff1
      value: 32.88192389840097
    - type: nauc_precision_at_3_max
      value: 47.956158764484755
    - type: nauc_precision_at_3_std
      value: -5.173137100242595
    - type: nauc_precision_at_5_diff1
      value: 22.380625664945793
    - type: nauc_precision_at_5_max
      value: 45.64181390208999
    - type: nauc_precision_at_5_std
      value: -0.7016753012171335
    - type: nauc_recall_at_1000_diff1
      value: 20.188527806658644
    - type: nauc_recall_at_1000_max
      value: 45.95015696119811
    - type: nauc_recall_at_1000_std
      value: 51.41886036852424
    - type: nauc_recall_at_100_diff1
      value: 27.997328229580198
    - type: nauc_recall_at_100_max
      value: 39.94640991165402
    - type: nauc_recall_at_100_std
      value: 19.365445255269094
    - type: nauc_recall_at_10_diff1
      value: 36.10982971358225
    - type: nauc_recall_at_10_max
      value: 36.02741554215535
    - type: nauc_recall_at_10_std
      value: -6.452154761173287
    - type: nauc_recall_at_1_diff1
      value: 46.62899301926139
    - type: nauc_recall_at_1_max
      value: 21.89247441846697
    - type: nauc_recall_at_1_std
      value: -14.202185813260924
    - type: nauc_recall_at_20_diff1
      value: 33.023788773735326
    - type: nauc_recall_at_20_max
      value: 37.341761583717364
    - type: nauc_recall_at_20_std
      value: 1.066929584398766
    - type: nauc_recall_at_3_diff1
      value: 41.0372289791838
    - type: nauc_recall_at_3_max
      value: 30.523122019252575
    - type: nauc_recall_at_3_std
      value: -11.873496181417412
    - type: nauc_recall_at_5_diff1
      value: 37.33321342892616
    - type: nauc_recall_at_5_max
      value: 32.17065703720826
    - type: nauc_recall_at_5_std
      value: -10.654073163768091
    - type: ndcg_at_1
      value: 54.321
    - type: ndcg_at_10
      value: 53.539
    - type: ndcg_at_100
      value: 60.080999999999996
    - type: ndcg_at_1000
      value: 62.141000000000005
    - type: ndcg_at_20
      value: 56.444
    - type: ndcg_at_3
      value: 49.753
    - type: ndcg_at_5
      value: 50.827999999999996
    - type: precision_at_1
      value: 54.321
    - type: precision_at_10
      value: 14.645
    - type: precision_at_100
      value: 2.153
    - type: precision_at_1000
      value: 0.252
    - type: precision_at_20
      value: 8.565000000000001
    - type: precision_at_3
      value: 32.922000000000004
    - type: precision_at_5
      value: 24.043
    - type: recall_at_1
      value: 27.778999999999996
    - type: recall_at_10
      value: 60.209
    - type: recall_at_100
      value: 83.73
    - type: recall_at_1000
      value: 96.036
    - type: recall_at_20
      value: 69.19999999999999
    - type: recall_at_3
      value: 44.79
    - type: recall_at_5
      value: 51.42
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB HotpotQA (default)
      revision: ab518f4d6fcca38d87c25209f94beba119d02014
      split: test
      type: mteb/hotpotqa
    metrics:
    - type: main_score
      value: 70.081
    - type: map_at_1
      value: 42.768
    - type: map_at_10
      value: 60.754
    - type: map_at_100
      value: 61.67700000000001
    - type: map_at_1000
      value: 61.736000000000004
    - type: map_at_20
      value: 61.312999999999995
    - type: map_at_3
      value: 57.607
    - type: map_at_5
      value: 59.573
    - type: mrr_at_1
      value: 85.53679945982444
    - type: mrr_at_10
      value: 89.72397993633645
    - type: mrr_at_100
      value: 89.83256060340038
    - type: mrr_at_1000
      value: 89.83589065514369
    - type: mrr_at_20
      value: 89.79433273696627
    - type: mrr_at_3
      value: 89.13796984019807
    - type: mrr_at_5
      value: 89.52487058293946
    - type: nauc_map_at_1000_diff1
      value: 6.895423973248177
    - type: nauc_map_at_1000_max
      value: 12.556131009800461
    - type: nauc_map_at_1000_std
      value: -4.399894824582004
    - type: nauc_map_at_100_diff1
      value: 6.850961524893442
    - type: nauc_map_at_100_max
      value: 12.527780526266113
    - type: nauc_map_at_100_std
      value: -4.39852354684375
    - type: nauc_map_at_10_diff1
      value: 6.924462849494474
    - type: nauc_map_at_10_max
      value: 12.617466119703954
    - type: nauc_map_at_10_std
      value: -5.1306322412222745
    - type: nauc_map_at_1_diff1
      value: 79.61545739962872
    - type: nauc_map_at_1_max
      value: 55.49501314532599
    - type: nauc_map_at_1_std
      value: -16.541819128254332
    - type: nauc_map_at_20_diff1
      value: 6.858106028657918
    - type: nauc_map_at_20_max
      value: 12.539408957471531
    - type: nauc_map_at_20_std
      value: -4.6716405210846546
    - type: nauc_map_at_3_diff1
      value: 9.317944310648734
    - type: nauc_map_at_3_max
      value: 14.209570427043802
    - type: nauc_map_at_3_std
      value: -6.6994640810793245
    - type: nauc_map_at_5_diff1
      value: 8.078948629915566
    - type: nauc_map_at_5_max
      value: 13.294246479999916
    - type: nauc_map_at_5_std
      value: -6.138839747619266
    - type: nauc_mrr_at_1000_diff1
      value: 79.58301401183212
    - type: nauc_mrr_at_1000_max
      value: 59.007684552475325
    - type: nauc_mrr_at_1000_std
      value: -14.787963814905044
    - type: nauc_mrr_at_100_diff1
      value: 79.5835776848335
    - type: nauc_mrr_at_100_max
      value: 59.0111617468661
    - type: nauc_mrr_at_100_std
      value: -14.788345203602649
    - type: nauc_mrr_at_10_diff1
      value: 79.62937026030107
    - type: nauc_mrr_at_10_max
      value: 59.18822318610967
    - type: nauc_mrr_at_10_std
      value: -14.783969129620134
    - type: nauc_mrr_at_1_diff1
      value: 79.61545739962872
    - type: nauc_mrr_at_1_max
      value: 55.49501314532599
    - type: nauc_mrr_at_1_std
      value: -16.541819128254332
    - type: nauc_mrr_at_20_diff1
      value: 79.60464151903025
    - type: nauc_mrr_at_20_max
      value: 59.0934141381388
    - type: nauc_mrr_at_20_std
      value: -14.75910060614095
    - type: nauc_mrr_at_3_diff1
      value: 79.3231509123134
    - type: nauc_mrr_at_3_max
      value: 59.10815442723091
    - type: nauc_mrr_at_3_std
      value: -15.150171022482631
    - type: nauc_mrr_at_5_diff1
      value: 79.56115133485501
    - type: nauc_mrr_at_5_max
      value: 59.30245170991139
    - type: nauc_mrr_at_5_std
      value: -14.900239877386662
    - type: nauc_ndcg_at_1000_diff1
      value: 11.693672871521773
    - type: nauc_ndcg_at_1000_max
      value: 15.837510906518714
    - type: nauc_ndcg_at_1000_std
      value: -1.8211019719566786
    - type: nauc_ndcg_at_100_diff1
      value: 10.42835912568642
    - type: nauc_ndcg_at_100_max
      value: 15.00709945900011
    - type: nauc_ndcg_at_100_std
      value: -1.5995456258147553
    - type: nauc_ndcg_at_10_diff1
      value: 10.982857146706952
    - type: nauc_ndcg_at_10_max
      value: 15.576099778424837
    - type: nauc_ndcg_at_10_std
      value: -4.449318507105899
    - type: nauc_ndcg_at_1_diff1
      value: 79.61545739962872
    - type: nauc_ndcg_at_1_max
      value: 55.49501314532599
    - type: nauc_ndcg_at_1_std
      value: -16.541819128254332
    - type: nauc_ndcg_at_20_diff1
      value: 10.528399976481227
    - type: nauc_ndcg_at_20_max
      value: 15.153669343696777
    - type: nauc_ndcg_at_20_std
      value: -3.1066308149357895
    - type: nauc_ndcg_at_3_diff1
      value: 15.68035303992369
    - type: nauc_ndcg_at_3_max
      value: 18.749350102632537
    - type: nauc_ndcg_at_3_std
      value: -7.185970853401351
    - type: nauc_ndcg_at_5_diff1
      value: 13.410571993832864
    - type: nauc_ndcg_at_5_max
      value: 17.1200043452206
    - type: nauc_ndcg_at_5_std
      value: -6.318836986249618
    - type: nauc_precision_at_1000_diff1
      value: -19.688446929534933
    - type: nauc_precision_at_1000_max
      value: -4.941241939653343
    - type: nauc_precision_at_1000_std
      value: 20.9661393157518
    - type: nauc_precision_at_100_diff1
      value: -18.508304896462853
    - type: nauc_precision_at_100_max
      value: -4.250205853424237
    - type: nauc_precision_at_100_std
      value: 15.235430975729619
    - type: nauc_precision_at_10_diff1
      value: -9.58790033346835
    - type: nauc_precision_at_10_max
      value: 2.419102858415595
    - type: nauc_precision_at_10_std
      value: 1.3354785196962335
    - type: nauc_precision_at_1_diff1
      value: 79.61545739962872
    - type: nauc_precision_at_1_max
      value: 55.49501314532599
    - type: nauc_precision_at_1_std
      value: -16.541819128254332
    - type: nauc_precision_at_20_diff1
      value: -12.541412085438669
    - type: nauc_precision_at_20_max
      value: 0.010826716292880342
    - type: nauc_precision_at_20_std
      value: 5.867977892873711
    - type: nauc_precision_at_3_diff1
      value: 0.9498347311360616
    - type: nauc_precision_at_3_max
      value: 9.908219933313948
    - type: nauc_precision_at_3_std
      value: -4.715300539821549
    - type: nauc_precision_at_5_diff1
      value: -3.321633551694524
    - type: nauc_precision_at_5_max
      value: 6.614956505743015
    - type: nauc_precision_at_5_std
      value: -3.090976099143575
    - type: nauc_recall_at_1000_diff1
      value: -19.688446929535118
    - type: nauc_recall_at_1000_max
      value: -4.941241939653247
    - type: nauc_recall_at_1000_std
      value: 20.966139315752123
    - type: nauc_recall_at_100_diff1
      value: -18.508304896462988
    - type: nauc_recall_at_100_max
      value: -4.250205853424454
    - type: nauc_recall_at_100_std
      value: 15.235430975729496
    - type: nauc_recall_at_10_diff1
      value: -9.587900333468165
    - type: nauc_recall_at_10_max
      value: 2.419102858415774
    - type: nauc_recall_at_10_std
      value: 1.3354785196963812
    - type: nauc_recall_at_1_diff1
      value: 79.61545739962872
    - type: nauc_recall_at_1_max
      value: 55.49501314532599
    - type: nauc_recall_at_1_std
      value: -16.541819128254332
    - type: nauc_recall_at_20_diff1
      value: -12.541412085438564
    - type: nauc_recall_at_20_max
      value: 0.01082671629291587
    - type: nauc_recall_at_20_std
      value: 5.867977892873685
    - type: nauc_recall_at_3_diff1
      value: 0.9498347311360453
    - type: nauc_recall_at_3_max
      value: 9.908219933313962
    - type: nauc_recall_at_3_std
      value: -4.715300539821613
    - type: nauc_recall_at_5_diff1
      value: -3.321633551694636
    - type: nauc_recall_at_5_max
      value: 6.614956505743056
    - type: nauc_recall_at_5_std
      value: -3.090976099143639
    - type: ndcg_at_1
      value: 85.53699999999999
    - type: ndcg_at_10
      value: 70.081
    - type: ndcg_at_100
      value: 73.096
    - type: ndcg_at_1000
      value: 74.22999999999999
    - type: ndcg_at_20
      value: 71.41
    - type: ndcg_at_3
      value: 65.835
    - type: ndcg_at_5
      value: 68.209
    - type: precision_at_1
      value: 85.53699999999999
    - type: precision_at_10
      value: 14.102999999999998
    - type: precision_at_100
      value: 1.644
    - type: precision_at_1000
      value: 0.179
    - type: precision_at_20
      value: 7.48
    - type: precision_at_3
      value: 40.778999999999996
    - type: precision_at_5
      value: 26.339000000000002
    - type: recall_at_1
      value: 42.768
    - type: recall_at_10
      value: 70.513
    - type: recall_at_100
      value: 82.194
    - type: recall_at_1000
      value: 89.71000000000001
    - type: recall_at_20
      value: 74.801
    - type: recall_at_3
      value: 61.168
    - type: recall_at_5
      value: 65.847
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ImdbClassification (default)
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
      split: test
      type: mteb/imdb
    metrics:
    - type: accuracy
      value: 95.18640000000002
    - type: ap
      value: 93.23333623278239
    - type: ap_weighted
      value: 93.23333623278239
    - type: f1
      value: 95.18544952344942
    - type: f1_weighted
      value: 95.18544952344945
    - type: main_score
      value: 95.18640000000002
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB MSMARCO (default)
      revision: c5a29a104738b98a9e76336939199e264163d4a0
      split: dev
      type: mteb/msmarco
    metrics:
    - type: main_score
      value: 42.171
    - type: map_at_1
      value: 22.395
    - type: map_at_10
      value: 35.038000000000004
    - type: map_at_100
      value: 36.205
    - type: map_at_1000
      value: 36.25
    - type: map_at_20
      value: 35.754000000000005
    - type: map_at_3
      value: 30.990000000000002
    - type: map_at_5
      value: 33.31
    - type: mrr_at_1
      value: 23.0945558739255
    - type: mrr_at_10
      value: 35.65352253604403
    - type: mrr_at_100
      value: 36.759941953109504
    - type: mrr_at_1000
      value: 36.79849525615509
    - type: mrr_at_20
      value: 36.33417189807629
    - type: mrr_at_3
      value: 31.67144221585482
    - type: mrr_at_5
      value: 33.96800382043935
    - type: nauc_map_at_1000_diff1
      value: 33.07894201425844
    - type: nauc_map_at_1000_max
      value: 5.497367123841249
    - type: nauc_map_at_1000_std
      value: -26.903002470094563
    - type: nauc_map_at_100_diff1
      value: 33.06682214144194
    - type: nauc_map_at_100_max
      value: 5.504009147391157
    - type: nauc_map_at_100_std
      value: -26.9011308052937
    - type: nauc_map_at_10_diff1
      value: 32.99762534741698
    - type: nauc_map_at_10_max
      value: 5.239485965024988
    - type: nauc_map_at_10_std
      value: -27.502688942110332
    - type: nauc_map_at_1_diff1
      value: 37.315440671209814
    - type: nauc_map_at_1_max
      value: 5.196967576959052
    - type: nauc_map_at_1_std
      value: -22.407538068541115
    - type: nauc_map_at_20_diff1
      value: 32.9852623735503
    - type: nauc_map_at_20_max
      value: 5.417945808988554
    - type: nauc_map_at_20_std
      value: -27.20254825981713
    - type: nauc_map_at_3_diff1
      value: 33.59004994989371
    - type: nauc_map_at_3_max
      value: 5.1103612164761625
    - type: nauc_map_at_3_std
      value: -26.618927641780942
    - type: nauc_map_at_5_diff1
      value: 33.08380356823687
    - type: nauc_map_at_5_max
      value: 5.17882180161743
    - type: nauc_map_at_5_std
      value: -27.266190392517757
    - type: nauc_mrr_at_1000_diff1
      value: 33.158621312069556
    - type: nauc_mrr_at_1000_max
      value: 5.672559589983072
    - type: nauc_mrr_at_1000_std
      value: -26.56447346785634
    - type: nauc_mrr_at_100_diff1
      value: 33.14697009719696
    - type: nauc_mrr_at_100_max
      value: 5.680814167599058
    - type: nauc_mrr_at_100_std
      value: -26.558304403534855
    - type: nauc_mrr_at_10_diff1
      value: 33.09205381282667
    - type: nauc_mrr_at_10_max
      value: 5.4683280580702505
    - type: nauc_mrr_at_10_std
      value: -27.110725288145794
    - type: nauc_mrr_at_1_diff1
      value: 37.42898316014141
    - type: nauc_mrr_at_1_max
      value: 5.29254880539881
    - type: nauc_mrr_at_1_std
      value: -22.498453357718958
    - type: nauc_mrr_at_20_diff1
      value: 33.044598141737126
    - type: nauc_mrr_at_20_max
      value: 5.620061958078301
    - type: nauc_mrr_at_20_std
      value: -26.80839803933076
    - type: nauc_mrr_at_3_diff1
      value: 33.684813089380796
    - type: nauc_mrr_at_3_max
      value: 5.28480339835537
    - type: nauc_mrr_at_3_std
      value: -26.396366697663627
    - type: nauc_mrr_at_5_diff1
      value: 33.14591564101392
    - type: nauc_mrr_at_5_max
      value: 5.435803588532608
    - type: nauc_mrr_at_5_std
      value: -26.9283416261166
    - type: nauc_ndcg_at_1000_diff1
      value: 32.26334751830434
    - type: nauc_ndcg_at_1000_max
      value: 6.398398814640009
    - type: nauc_ndcg_at_1000_std
      value: -26.41366724776481
    - type: nauc_ndcg_at_100_diff1
      value: 31.939387581789603
    - type: nauc_ndcg_at_100_max
      value: 6.734240408227785
    - type: nauc_ndcg_at_100_std
      value: -26.071672454601234
    - type: nauc_ndcg_at_10_diff1
      value: 31.465010764110524
    - type: nauc_ndcg_at_10_max
      value: 5.503472581176608
    - type: nauc_ndcg_at_10_std
      value: -29.33683329617212
    - type: nauc_ndcg_at_1_diff1
      value: 37.42898316014141
    - type: nauc_ndcg_at_1_max
      value: 5.29254880539881
    - type: nauc_ndcg_at_1_std
      value: -22.498453357718958
    - type: nauc_ndcg_at_20_diff1
      value: 31.305511008884636
    - type: nauc_ndcg_at_20_max
      value: 6.102297303077842
    - type: nauc_ndcg_at_20_std
      value: -28.257988389916473
    - type: nauc_ndcg_at_3_diff1
      value: 32.642328616979036
    - type: nauc_ndcg_at_3_max
      value: 5.184277916519752
    - type: nauc_ndcg_at_3_std
      value: -27.675255428080973
    - type: nauc_ndcg_at_5_diff1
      value: 31.652750958445896
    - type: nauc_ndcg_at_5_max
      value: 5.385050705719649
    - type: nauc_ndcg_at_5_std
      value: -28.76184758964142
    - type: nauc_precision_at_1000_diff1
      value: -0.3891260906731712
    - type: nauc_precision_at_1000_max
      value: 14.654890907611593
    - type: nauc_precision_at_1000_std
      value: 16.015857041438103
    - type: nauc_precision_at_100_diff1
      value: 12.552710496326661
    - type: nauc_precision_at_100_max
      value: 17.62855649363789
    - type: nauc_precision_at_100_std
      value: 2.2564523950113413
    - type: nauc_precision_at_10_diff1
      value: 24.027047781498133
    - type: nauc_precision_at_10_max
      value: 6.7973552756967255
    - type: nauc_precision_at_10_std
      value: -32.69204404285194
    - type: nauc_precision_at_1_diff1
      value: 37.42898316014141
    - type: nauc_precision_at_1_max
      value: 5.29254880539881
    - type: nauc_precision_at_1_std
      value: -22.498453357718958
    - type: nauc_precision_at_20_diff1
      value: 20.161232545534403
    - type: nauc_precision_at_20_max
      value: 9.925718004119801
    - type: nauc_precision_at_20_std
      value: -25.926363290049114
    - type: nauc_precision_at_3_diff1
      value: 29.531999144163578
    - type: nauc_precision_at_3_max
      value: 5.359828933652676
    - type: nauc_precision_at_3_std
      value: -30.2540345351995
    - type: nauc_precision_at_5_diff1
      value: 26.727388439013122
    - type: nauc_precision_at_5_max
      value: 6.007786981571831
    - type: nauc_precision_at_5_std
      value: -32.17049672640078
    - type: nauc_recall_at_1000_diff1
      value: 28.187251832683284
    - type: nauc_recall_at_1000_max
      value: 43.45387520636765
    - type: nauc_recall_at_1000_std
      value: 39.10855230211973
    - type: nauc_recall_at_100_diff1
      value: 24.108737692339275
    - type: nauc_recall_at_100_max
      value: 21.079134312567312
    - type: nauc_recall_at_100_std
      value: -5.565359235895099
    - type: nauc_recall_at_10_diff1
      value: 25.983903477459187
    - type: nauc_recall_at_10_max
      value: 5.966150803468692
    - type: nauc_recall_at_10_std
      value: -35.586299027302566
    - type: nauc_recall_at_1_diff1
      value: 37.315440671209814
    - type: nauc_recall_at_1_max
      value: 5.196967576959052
    - type: nauc_recall_at_1_std
      value: -22.407538068541115
    - type: nauc_recall_at_20_diff1
      value: 23.783547588059438
    - type: nauc_recall_at_20_max
      value: 8.852469804491404
    - type: nauc_recall_at_20_std
      value: -31.899893515192556
    - type: nauc_recall_at_3_diff1
      value: 29.829493798520083
    - type: nauc_recall_at_3_max
      value: 5.201295571380362
    - type: nauc_recall_at_3_std
      value: -30.423636702611645
    - type: nauc_recall_at_5_diff1
      value: 27.189713030747058
    - type: nauc_recall_at_5_max
      value: 5.678448934443972
    - type: nauc_recall_at_5_std
      value: -32.96654981430625
    - type: ndcg_at_1
      value: 23.095
    - type: ndcg_at_10
      value: 42.171
    - type: ndcg_at_100
      value: 47.77
    - type: ndcg_at_1000
      value: 48.862
    - type: ndcg_at_20
      value: 44.716
    - type: ndcg_at_3
      value: 33.924
    - type: ndcg_at_5
      value: 38.061
    - type: precision_at_1
      value: 23.095
    - type: precision_at_10
      value: 6.712999999999999
    - type: precision_at_100
      value: 0.951
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_20
      value: 3.891
    - type: precision_at_3
      value: 14.456
    - type: precision_at_5
      value: 10.774000000000001
    - type: recall_at_1
      value: 22.395
    - type: recall_at_10
      value: 64.16
    - type: recall_at_100
      value: 89.97500000000001
    - type: recall_at_1000
      value: 98.275
    - type: recall_at_20
      value: 74.054
    - type: recall_at_3
      value: 41.739
    - type: recall_at_5
      value: 51.662
    task:
      type: Retrieval
  - dataset:
      config: en
      name: MTEB MTOPDomainClassification (en)
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
      split: test
      type: mteb/mtop_domain
    metrics:
    - type: accuracy
      value: 97.17738258093934
    - type: f1
      value: 96.92736627026595
    - type: f1_weighted
      value: 97.17553826461106
    - type: main_score
      value: 97.17738258093934
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB MTOPIntentClassification (en)
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
      split: test
      type: mteb/mtop_intent
    metrics:
    - type: accuracy
      value: 93.54081167350661
    - type: f1
      value: 79.49055985796305
    - type: f1_weighted
      value: 94.44639337688237
    - type: main_score
      value: 93.54081167350661
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB MassiveIntentClassification (en)
      revision: 4672e20407010da34463acc759c162ca9734bca6
      split: test
      type: mteb/amazon_massive_intent
    metrics:
    - type: accuracy
      value: 77.4747814391392
    - type: f1
      value: 75.37170868139704
    - type: f1_weighted
      value: 77.52688280321094
    - type: main_score
      value: 77.4747814391392
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB MassiveScenarioClassification (en)
      revision: fad2c6e8459f9e1c45d9315f4953d921437d70f8
      split: test
      type: mteb/amazon_massive_scenario
    metrics:
    - type: accuracy
      value: 78.70544720914593
    - type: f1
      value: 78.22460725260667
    - type: f1_weighted
      value: 78.62916991376048
    - type: main_score
      value: 78.70544720914593
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB MedrxivClusteringP2P (default)
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
      split: test
      type: mteb/medrxiv-clustering-p2p
    metrics:
    - type: main_score
      value: 42.69488311001275
    - type: v_measure
      value: 42.69488311001275
    - type: v_measure_std
      value: 1.082652834981339
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB MedrxivClusteringS2S (default)
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
      split: test
      type: mteb/medrxiv-clustering-s2s
    metrics:
    - type: main_score
      value: 40.692196438019636
    - type: v_measure
      value: 40.692196438019636
    - type: v_measure_std
      value: 1.480857038869457
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB MindSmallReranking (default)
      revision: 59042f120c80e8afa9cdbb224f67076cec0fc9a7
      split: test
      type: mteb/mind_small
    metrics:
    - type: main_score
      value: 31.740632153828336
    - type: map
      value: 31.740632153828336
    - type: mrr
      value: 32.939110813064126
    - type: nAUC_map_diff1
      value: 8.85094587186729
    - type: nAUC_map_max
      value: -20.53497314931837
    - type: nAUC_map_std
      value: -3.515268578344146
    - type: nAUC_mrr_diff1
      value: 8.602415649120731
    - type: nAUC_mrr_max
      value: -15.149944382405678
    - type: nAUC_mrr_std
      value: -1.3268027458327625
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB NFCorpus (default)
      revision: ec0fa4fe99da2ff19ca1214b7966684033a58814
      split: test
      type: mteb/nfcorpus
    metrics:
    - type: main_score
      value: 37.966
    - type: map_at_1
      value: 6.5329999999999995
    - type: map_at_10
      value: 14.197000000000001
    - type: map_at_100
      value: 18.21
    - type: map_at_1000
      value: 19.753999999999998
    - type: map_at_20
      value: 15.833
    - type: map_at_3
      value: 10.524000000000001
    - type: map_at_5
      value: 12.19
    - type: mrr_at_1
      value: 52.012383900928796
    - type: mrr_at_10
      value: 59.955034645437124
    - type: mrr_at_100
      value: 60.42427452831739
    - type: mrr_at_1000
      value: 60.45665318830144
    - type: mrr_at_20
      value: 60.26579513285032
    - type: mrr_at_3
      value: 57.84313725490197
    - type: mrr_at_5
      value: 59.050567595459235
    - type: nauc_map_at_1000_diff1
      value: 22.570172161818135
    - type: nauc_map_at_1000_max
      value: 33.38706056795645
    - type: nauc_map_at_1000_std
      value: 13.670500561336912
    - type: nauc_map_at_100_diff1
      value: 23.634919572912576
    - type: nauc_map_at_100_max
      value: 32.14677377697161
    - type: nauc_map_at_100_std
      value: 9.703024063294267
    - type: nauc_map_at_10_diff1
      value: 29.343647622403136
    - type: nauc_map_at_10_max
      value: 27.591406933508107
    - type: nauc_map_at_10_std
      value: -2.0367614601960913
    - type: nauc_map_at_1_diff1
      value: 44.42122271802935
    - type: nauc_map_at_1_max
      value: 12.78511682175368
    - type: nauc_map_at_1_std
      value: -14.830233712927019
    - type: nauc_map_at_20_diff1
      value: 26.511043382640715
    - type: nauc_map_at_20_max
      value: 29.462049542564
    - type: nauc_map_at_20_std
      value: 2.8044331075339954
    - type: nauc_map_at_3_diff1
      value: 36.806244382659884
    - type: nauc_map_at_3_max
      value: 20.987190477461933
    - type: nauc_map_at_3_std
      value: -9.054553004626698
    - type: nauc_map_at_5_diff1
      value: 32.31717753549589
    - type: nauc_map_at_5_max
      value: 24.479325198371065
    - type: nauc_map_at_5_std
      value: -6.342215509205215
    - type: nauc_mrr_at_1000_diff1
      value: 34.11627162920768
    - type: nauc_mrr_at_1000_max
      value: 49.06442028782317
    - type: nauc_mrr_at_1000_std
      value: 30.163589739210927
    - type: nauc_mrr_at_100_diff1
      value: 34.107897934881734
    - type: nauc_mrr_at_100_max
      value: 49.0818796462814
    - type: nauc_mrr_at_100_std
      value: 30.18393167110018
    - type: nauc_mrr_at_10_diff1
      value: 34.630453182235584
    - type: nauc_mrr_at_10_max
      value: 49.00517141658673
    - type: nauc_mrr_at_10_std
      value: 29.85566427285698
    - type: nauc_mrr_at_1_diff1
      value: 33.63132944262702
    - type: nauc_mrr_at_1_max
      value: 44.7696536853246
    - type: nauc_mrr_at_1_std
      value: 24.55895319594289
    - type: nauc_mrr_at_20_diff1
      value: 34.16473341371902
    - type: nauc_mrr_at_20_max
      value: 49.14602047626697
    - type: nauc_mrr_at_20_std
      value: 30.249090566290153
    - type: nauc_mrr_at_3_diff1
      value: 34.56482749158797
    - type: nauc_mrr_at_3_max
      value: 47.1046107449135
    - type: nauc_mrr_at_3_std
      value: 27.764607001025883
    - type: nauc_mrr_at_5_diff1
      value: 34.52298637250081
    - type: nauc_mrr_at_5_max
      value: 48.732533656670576
    - type: nauc_mrr_at_5_std
      value: 29.223042020025254
    - type: nauc_ndcg_at_1000_diff1
      value: 21.147261019987404
    - type: nauc_ndcg_at_1000_max
      value: 49.178411796523925
    - type: nauc_ndcg_at_1000_std
      value: 34.38707710634909
    - type: nauc_ndcg_at_100_diff1
      value: 20.490187383178775
    - type: nauc_ndcg_at_100_max
      value: 43.05738648001313
    - type: nauc_ndcg_at_100_std
      value: 28.070322090677536
    - type: nauc_ndcg_at_10_diff1
      value: 17.96864647514039
    - type: nauc_ndcg_at_10_max
      value: 43.145412626878134
    - type: nauc_ndcg_at_10_std
      value: 28.83163319404016
    - type: nauc_ndcg_at_1_diff1
      value: 32.535179808378366
    - type: nauc_ndcg_at_1_max
      value: 44.34101276733621
    - type: nauc_ndcg_at_1_std
      value: 24.669618799727537
    - type: nauc_ndcg_at_20_diff1
      value: 17.21778534080309
    - type: nauc_ndcg_at_20_max
      value: 40.81778538465099
    - type: nauc_ndcg_at_20_std
      value: 28.665856350810493
    - type: nauc_ndcg_at_3_diff1
      value: 22.77038520233727
    - type: nauc_ndcg_at_3_max
      value: 44.30289584470762
    - type: nauc_ndcg_at_3_std
      value: 26.22513862717806
    - type: nauc_ndcg_at_5_diff1
      value: 20.183649681788783
    - type: nauc_ndcg_at_5_max
      value: 45.1906910505147
    - type: nauc_ndcg_at_5_std
      value: 27.050998277816596
    - type: nauc_precision_at_1000_diff1
      value: -9.637582115680226
    - type: nauc_precision_at_1000_max
      value: 12.262807708348582
    - type: nauc_precision_at_1000_std
      value: 39.21228873334985
    - type: nauc_precision_at_100_diff1
      value: -12.148383949107888
    - type: nauc_precision_at_100_max
      value: 22.520368329128875
    - type: nauc_precision_at_100_std
      value: 46.80957575009526
    - type: nauc_precision_at_10_diff1
      value: -1.4514982237388117
    - type: nauc_precision_at_10_max
      value: 39.45063449321164
    - type: nauc_precision_at_10_std
      value: 40.86973125094316
    - type: nauc_precision_at_1_diff1
      value: 34.42233528833798
    - type: nauc_precision_at_1_max
      value: 45.12368103749432
    - type: nauc_precision_at_1_std
      value: 24.978070230808363
    - type: nauc_precision_at_20_diff1
      value: -6.510566965012683
    - type: nauc_precision_at_20_max
      value: 32.53639296500517
    - type: nauc_precision_at_20_std
      value: 44.460925236966254
    - type: nauc_precision_at_3_diff1
      value: 11.603111986019602
    - type: nauc_precision_at_3_max
      value: 44.00325236514464
    - type: nauc_precision_at_3_std
      value: 31.24949711766119
    - type: nauc_precision_at_5_diff1
      value: 5.138531027391432
    - type: nauc_precision_at_5_max
      value: 43.9713097240123
    - type: nauc_precision_at_5_std
      value: 34.2875261139779
    - type: nauc_recall_at_1000_diff1
      value: 5.552109384812395
    - type: nauc_recall_at_1000_max
      value: 27.235866447040923
    - type: nauc_recall_at_1000_std
      value: 22.825381898683002
    - type: nauc_recall_at_100_diff1
      value: 11.223120096283072
    - type: nauc_recall_at_100_max
      value: 25.018619163376933
    - type: nauc_recall_at_100_std
      value: 13.254885339665625
    - type: nauc_recall_at_10_diff1
      value: 23.291157649666474
    - type: nauc_recall_at_10_max
      value: 22.594976294724926
    - type: nauc_recall_at_10_std
      value: -3.3745087550094652
    - type: nauc_recall_at_1_diff1
      value: 44.42122271802935
    - type: nauc_recall_at_1_max
      value: 12.78511682175368
    - type: nauc_recall_at_1_std
      value: -14.830233712927019
    - type: nauc_recall_at_20_diff1
      value: 16.227917996924248
    - type: nauc_recall_at_20_max
      value: 21.73111226469191
    - type: nauc_recall_at_20_std
      value: 1.1151296835950228
    - type: nauc_recall_at_3_diff1
      value: 34.30285111001446
    - type: nauc_recall_at_3_max
      value: 19.195888942769514
    - type: nauc_recall_at_3_std
      value: -9.160448072987153
    - type: nauc_recall_at_5_diff1
      value: 28.51210336037057
    - type: nauc_recall_at_5_max
      value: 23.33761584690785
    - type: nauc_recall_at_5_std
      value: -5.320194496450298
    - type: ndcg_at_1
      value: 50.31
    - type: ndcg_at_10
      value: 37.966
    - type: ndcg_at_100
      value: 34.92
    - type: ndcg_at_1000
      value: 43.391000000000005
    - type: ndcg_at_20
      value: 35.598
    - type: ndcg_at_3
      value: 44.533
    - type: ndcg_at_5
      value: 41.492000000000004
    - type: precision_at_1
      value: 51.702999999999996
    - type: precision_at_10
      value: 27.802
    - type: precision_at_100
      value: 8.770999999999999
    - type: precision_at_1000
      value: 2.16
    - type: precision_at_20
      value: 20.774
    - type: precision_at_3
      value: 41.589
    - type: precision_at_5
      value: 35.356
    - type: recall_at_1
      value: 6.5329999999999995
    - type: recall_at_10
      value: 18.089
    - type: recall_at_100
      value: 34.892
    - type: recall_at_1000
      value: 65.422
    - type: recall_at_20
      value: 22.431
    - type: recall_at_3
      value: 11.417
    - type: recall_at_5
      value: 13.986
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB NQ (default)
      revision: b774495ed302d8c44a3a7ea25c90dbce03968f31
      split: test
      type: mteb/nq
    metrics:
    - type: main_score
      value: 65.379
    - type: map_at_1
      value: 41.0
    - type: map_at_10
      value: 58.004
    - type: map_at_100
      value: 58.711
    - type: map_at_1000
      value: 58.727
    - type: map_at_20
      value: 58.497
    - type: map_at_3
      value: 54.092
    - type: map_at_5
      value: 56.455
    - type: mrr_at_1
      value: 46.00231749710313
    - type: mrr_at_10
      value: 60.42321911383325
    - type: mrr_at_100
      value: 60.877849730039976
    - type: mrr_at_1000
      value: 60.88836542473732
    - type: mrr_at_20
      value: 60.73386912832637
    - type: mrr_at_3
      value: 57.536693704132865
    - type: mrr_at_5
      value: 59.35303205870993
    - type: nauc_map_at_1000_diff1
      value: 39.01637329819782
    - type: nauc_map_at_1000_max
      value: 30.235904669554724
    - type: nauc_map_at_1000_std
      value: -5.5376982523462495
    - type: nauc_map_at_100_diff1
      value: 39.01473909268306
    - type: nauc_map_at_100_max
      value: 30.256923402388917
    - type: nauc_map_at_100_std
      value: -5.521375627438218
    - type: nauc_map_at_10_diff1
      value: 38.90277380530629
    - type: nauc_map_at_10_max
      value: 30.297309844026653
    - type: nauc_map_at_10_std
      value: -5.817825780032027
    - type: nauc_map_at_1_diff1
      value: 42.32788852329057
    - type: nauc_map_at_1_max
      value: 24.46544336510515
    - type: nauc_map_at_1_std
      value: -7.520423929986615
    - type: nauc_map_at_20_diff1
      value: 38.97689805011414
    - type: nauc_map_at_20_max
      value: 30.298002095016603
    - type: nauc_map_at_20_std
      value: -5.575771779362985
    - type: nauc_map_at_3_diff1
      value: 38.674219941864166
    - type: nauc_map_at_3_max
      value: 29.27040971284196
    - type: nauc_map_at_3_std
      value: -6.959754940595987
    - type: nauc_map_at_5_diff1
      value: 38.48505963388823
    - type: nauc_map_at_5_max
      value: 29.709676184981106
    - type: nauc_map_at_5_std
      value: -6.647064377327948
    - type: nauc_mrr_at_1000_diff1
      value: 39.32103909244816
    - type: nauc_mrr_at_1000_max
      value: 31.167694787108964
    - type: nauc_mrr_at_1000_std
      value: -3.9238634506559964
    - type: nauc_mrr_at_100_diff1
      value: 39.32056263149422
    - type: nauc_mrr_at_100_max
      value: 31.181310756322816
    - type: nauc_mrr_at_100_std
      value: -3.9128237479050143
    - type: nauc_mrr_at_10_diff1
      value: 39.20347890809096
    - type: nauc_mrr_at_10_max
      value: 31.303682649274528
    - type: nauc_mrr_at_10_std
      value: -4.024398873646956
    - type: nauc_mrr_at_1_diff1
      value: 42.35750570597814
    - type: nauc_mrr_at_1_max
      value: 27.001164454337417
    - type: nauc_mrr_at_1_std
      value: -5.4425261166746175
    - type: nauc_mrr_at_20_diff1
      value: 39.29430685185283
    - type: nauc_mrr_at_20_max
      value: 31.261259849521366
    - type: nauc_mrr_at_20_std
      value: -3.8721731067039724
    - type: nauc_mrr_at_3_diff1
      value: 39.02054328436386
    - type: nauc_mrr_at_3_max
      value: 30.974711774404383
    - type: nauc_mrr_at_3_std
      value: -4.3999930344506515
    - type: nauc_mrr_at_5_diff1
      value: 38.75751629287903
    - type: nauc_mrr_at_5_max
      value: 31.132661040406816
    - type: nauc_mrr_at_5_std
      value: -4.29840668391511
    - type: nauc_ndcg_at_1000_diff1
      value: 38.698208469095555
    - type: nauc_ndcg_at_1000_max
      value: 31.978150517678216
    - type: nauc_ndcg_at_1000_std
      value: -3.737573848984746
    - type: nauc_ndcg_at_100_diff1
      value: 38.67266393694767
    - type: nauc_ndcg_at_100_max
      value: 32.522791133580974
    - type: nauc_ndcg_at_100_std
      value: -3.2282327394771584
    - type: nauc_ndcg_at_10_diff1
      value: 38.21749997117465
    - type: nauc_ndcg_at_10_max
      value: 32.95900249159679
    - type: nauc_ndcg_at_10_std
      value: -4.200011411678329
    - type: nauc_ndcg_at_1_diff1
      value: 42.43479355183211
    - type: nauc_ndcg_at_1_max
      value: 26.98449556778994
    - type: nauc_ndcg_at_1_std
      value: -5.371110571047643
    - type: nauc_ndcg_at_20_diff1
      value: 38.48289744588944
    - type: nauc_ndcg_at_20_max
      value: 33.00256819622145
    - type: nauc_ndcg_at_20_std
      value: -3.2767517372363955
    - type: nauc_ndcg_at_3_diff1
      value: 37.766963348083436
    - type: nauc_ndcg_at_3_max
      value: 31.076996255111105
    - type: nauc_ndcg_at_3_std
      value: -6.2575061269806564
    - type: nauc_ndcg_at_5_diff1
      value: 37.237416210455365
    - type: nauc_ndcg_at_5_max
      value: 31.75980002710163
    - type: nauc_ndcg_at_5_std
      value: -5.852513629179639
    - type: nauc_precision_at_1000_diff1
      value: -9.498117144733484
    - type: nauc_precision_at_1000_max
      value: 4.3890953547264075
    - type: nauc_precision_at_1000_std
      value: 14.63293972919392
    - type: nauc_precision_at_100_diff1
      value: -5.924747900849213
    - type: nauc_precision_at_100_max
      value: 10.646106594305916
    - type: nauc_precision_at_100_std
      value: 17.14170198738387
    - type: nauc_precision_at_10_diff1
      value: 7.386946930660766
    - type: nauc_precision_at_10_max
      value: 23.32138059612198
    - type: nauc_precision_at_10_std
      value: 10.407837742465585
    - type: nauc_precision_at_1_diff1
      value: 42.43479355183211
    - type: nauc_precision_at_1_max
      value: 26.98449556778994
    - type: nauc_precision_at_1_std
      value: -5.371110571047643
    - type: nauc_precision_at_20_diff1
      value: 1.7277447209275587
    - type: nauc_precision_at_20_max
      value: 18.673328501447468
    - type: nauc_precision_at_20_std
      value: 14.896062507561728
    - type: nauc_precision_at_3_diff1
      value: 21.70545024457268
    - type: nauc_precision_at_3_max
      value: 30.146918090108937
    - type: nauc_precision_at_3_std
      value: -0.10890220706575011
    - type: nauc_precision_at_5_diff1
      value: 13.302161930674568
    - type: nauc_precision_at_5_max
      value: 26.160925964700926
    - type: nauc_precision_at_5_std
      value: 3.085490713919532
    - type: nauc_recall_at_1000_diff1
      value: 20.140389098840384
    - type: nauc_recall_at_1000_max
      value: 73.50267480088016
    - type: nauc_recall_at_1000_std
      value: 55.778759998421116
    - type: nauc_recall_at_100_diff1
      value: 32.69109764534067
    - type: nauc_recall_at_100_max
      value: 68.51246466609874
    - type: nauc_recall_at_100_std
      value: 37.218384918904164
    - type: nauc_recall_at_10_diff1
      value: 32.773742201577946
    - type: nauc_recall_at_10_max
      value: 44.90339100321307
    - type: nauc_recall_at_10_std
      value: 1.5439377311389577
    - type: nauc_recall_at_1_diff1
      value: 42.32788852329057
    - type: nauc_recall_at_1_max
      value: 24.46544336510515
    - type: nauc_recall_at_1_std
      value: -7.520423929986615
    - type: nauc_recall_at_20_diff1
      value: 33.55395190171798
    - type: nauc_recall_at_20_max
      value: 51.717667740401595
    - type: nauc_recall_at_20_std
      value: 12.392156763343655
    - type: nauc_recall_at_3_diff1
      value: 33.3222578317937
    - type: nauc_recall_at_3_max
      value: 34.20178017207147
    - type: nauc_recall_at_3_std
      value: -6.4347188425645925
    - type: nauc_recall_at_5_diff1
      value: 30.558717837909633
    - type: nauc_recall_at_5_max
      value: 36.641376929675005
    - type: nauc_recall_at_5_std
      value: -5.818388719321608
    - type: ndcg_at_1
      value: 45.973000000000006
    - type: ndcg_at_10
      value: 65.379
    - type: ndcg_at_100
      value: 68.125
    - type: ndcg_at_1000
      value: 68.474
    - type: ndcg_at_20
      value: 66.873
    - type: ndcg_at_3
      value: 58.42
    - type: ndcg_at_5
      value: 62.197
    - type: precision_at_1
      value: 45.973000000000006
    - type: precision_at_10
      value: 10.2
    - type: precision_at_100
      value: 1.176
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_20
      value: 5.473999999999999
    - type: precision_at_3
      value: 26.255
    - type: precision_at_5
      value: 18.001
    - type: recall_at_1
      value: 41.0
    - type: recall_at_10
      value: 85.04
    - type: recall_at_100
      value: 96.637
    - type: recall_at_1000
      value: 99.223
    - type: recall_at_20
      value: 90.464
    - type: recall_at_3
      value: 67.44200000000001
    - type: recall_at_5
      value: 75.992
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB QuoraRetrieval (default)
      revision: e4e08e0b7dbe3c8700f0daef558ff32256715259
      split: test
      type: mteb/quora
    metrics:
    - type: main_score
      value: 90.361
    - type: map_at_1
      value: 72.68
    - type: map_at_10
      value: 86.98599999999999
    - type: map_at_100
      value: 87.58399999999999
    - type: map_at_1000
      value: 87.595
    - type: map_at_20
      value: 87.392
    - type: map_at_3
      value: 84.107
    - type: map_at_5
      value: 85.963
    - type: mrr_at_1
      value: 83.76
    - type: mrr_at_10
      value: 89.54791666666667
    - type: mrr_at_100
      value: 89.61443180110122
    - type: mrr_at_1000
      value: 89.61504234132673
    - type: mrr_at_20
      value: 89.60038441275788
    - type: mrr_at_3
      value: 88.77666666666666
    - type: mrr_at_5
      value: 89.32916666666667
    - type: nauc_map_at_1000_diff1
      value: 78.64561944799313
    - type: nauc_map_at_1000_max
      value: 35.65116972441742
    - type: nauc_map_at_1000_std
      value: -59.176504681214745
    - type: nauc_map_at_100_diff1
      value: 78.64981229211423
    - type: nauc_map_at_100_max
      value: 35.634529287549256
    - type: nauc_map_at_100_std
      value: -59.243321003031355
    - type: nauc_map_at_10_diff1
      value: 78.7723509153348
    - type: nauc_map_at_10_max
      value: 35.13537410840319
    - type: nauc_map_at_10_std
      value: -61.68050482817051
    - type: nauc_map_at_1_diff1
      value: 82.62919644579914
    - type: nauc_map_at_1_max
      value: 26.04586082718353
    - type: nauc_map_at_1_std
      value: -50.42816921722124
    - type: nauc_map_at_20_diff1
      value: 78.71225141233032
    - type: nauc_map_at_20_max
      value: 35.49883905797678
    - type: nauc_map_at_20_std
      value: -60.35209557648571
    - type: nauc_map_at_3_diff1
      value: 79.31557965102503
    - type: nauc_map_at_3_max
      value: 32.15058687859126
    - type: nauc_map_at_3_std
      value: -63.26151805019198
    - type: nauc_map_at_5_diff1
      value: 78.99230629909208
    - type: nauc_map_at_5_max
      value: 33.9472078572996
    - type: nauc_map_at_5_std
      value: -63.27892765966846
    - type: nauc_mrr_at_1000_diff1
      value: 79.49639701960734
    - type: nauc_mrr_at_1000_max
      value: 37.5898040965896
    - type: nauc_mrr_at_1000_std
      value: -55.50965805943409
    - type: nauc_mrr_at_100_diff1
      value: 79.49674894854407
    - type: nauc_mrr_at_100_max
      value: 37.591403280981936
    - type: nauc_mrr_at_100_std
      value: -55.51279091639145
    - type: nauc_mrr_at_10_diff1
      value: 79.49489661505308
    - type: nauc_mrr_at_10_max
      value: 37.662907518382454
    - type: nauc_mrr_at_10_std
      value: -55.6727807317756
    - type: nauc_mrr_at_1_diff1
      value: 80.13313405297028
    - type: nauc_mrr_at_1_max
      value: 37.46885487643649
    - type: nauc_mrr_at_1_std
      value: -51.840952235669
    - type: nauc_mrr_at_20_diff1
      value: 79.49774661144632
    - type: nauc_mrr_at_20_max
      value: 37.61017252546323
    - type: nauc_mrr_at_20_std
      value: -55.55908923667907
    - type: nauc_mrr_at_3_diff1
      value: 79.26661222088983
    - type: nauc_mrr_at_3_max
      value: 37.40830620396969
    - type: nauc_mrr_at_3_std
      value: -56.2328472062151
    - type: nauc_mrr_at_5_diff1
      value: 79.40150849185312
    - type: nauc_mrr_at_5_max
      value: 37.62921214147551
    - type: nauc_mrr_at_5_std
      value: -55.99061561433468
    - type: nauc_ndcg_at_1000_diff1
      value: 78.63912486840212
    - type: nauc_ndcg_at_1000_max
      value: 37.074671305868364
    - type: nauc_ndcg_at_1000_std
      value: -57.03128439727864
    - type: nauc_ndcg_at_100_diff1
      value: 78.66279464965955
    - type: nauc_ndcg_at_100_max
      value: 37.04194410026304
    - type: nauc_ndcg_at_100_std
      value: -57.3588663941289
    - type: nauc_ndcg_at_10_diff1
      value: 78.66174243836718
    - type: nauc_ndcg_at_10_max
      value: 36.30923406132181
    - type: nauc_ndcg_at_10_std
      value: -61.397213215672174
    - type: nauc_ndcg_at_1_diff1
      value: 80.09374685621687
    - type: nauc_ndcg_at_1_max
      value: 37.74836644481908
    - type: nauc_ndcg_at_1_std
      value: -51.61176838484393
    - type: nauc_ndcg_at_20_diff1
      value: 78.73251657218076
    - type: nauc_ndcg_at_20_max
      value: 36.639768285510605
    - type: nauc_ndcg_at_20_std
      value: -59.78324743396675
    - type: nauc_ndcg_at_3_diff1
      value: 77.86646091716895
    - type: nauc_ndcg_at_3_max
      value: 34.55217778682923
    - type: nauc_ndcg_at_3_std
      value: -61.47162812597281
    - type: nauc_ndcg_at_5_diff1
      value: 78.34185599308286
    - type: nauc_ndcg_at_5_max
      value: 35.39283466232233
    - type: nauc_ndcg_at_5_std
      value: -62.540843030282154
    - type: nauc_precision_at_1000_diff1
      value: -46.03451028926803
    - type: nauc_precision_at_1000_max
      value: -6.392795758540018
    - type: nauc_precision_at_1000_std
      value: 47.88292950567429
    - type: nauc_precision_at_100_diff1
      value: -45.77947437539288
    - type: nauc_precision_at_100_max
      value: -6.218521423996721
    - type: nauc_precision_at_100_std
      value: 46.35155740207614
    - type: nauc_precision_at_10_diff1
      value: -42.51402319644932
    - type: nauc_precision_at_10_max
      value: -2.1142563863777633
    - type: nauc_precision_at_10_std
      value: 31.409102250615195
    - type: nauc_precision_at_1_diff1
      value: 80.09374685621687
    - type: nauc_precision_at_1_max
      value: 37.74836644481908
    - type: nauc_precision_at_1_std
      value: -51.61176838484393
    - type: nauc_precision_at_20_diff1
      value: -44.60955088543413
    - type: nauc_precision_at_20_max
      value: -4.203274111814019
    - type: nauc_precision_at_20_std
      value: 38.57979281558364
    - type: nauc_precision_at_3_diff1
      value: -25.76366107755223
    - type: nauc_precision_at_3_max
      value: 5.574991906763969
    - type: nauc_precision_at_3_std
      value: 6.638992632634816
    - type: nauc_precision_at_5_diff1
      value: -36.45978749108418
    - type: nauc_precision_at_5_max
      value: 1.1774897012321879
    - type: nauc_precision_at_5_std
      value: 19.841595721112153
    - type: nauc_recall_at_1000_diff1
      value: 64.05097840215399
    - type: nauc_recall_at_1000_max
      value: -3.954743235786646
    - type: nauc_recall_at_1000_std
      value: -32.639442017333934
    - type: nauc_recall_at_100_diff1
      value: 77.44679532176043
    - type: nauc_recall_at_100_max
      value: 42.57470511863274
    - type: nauc_recall_at_100_std
      value: -85.09515908249318
    - type: nauc_recall_at_10_diff1
      value: 75.56303387377868
    - type: nauc_recall_at_10_max
      value: 33.5975131408819
    - type: nauc_recall_at_10_std
      value: -91.86562605019546
    - type: nauc_recall_at_1_diff1
      value: 82.62919644579914
    - type: nauc_recall_at_1_max
      value: 26.04586082718353
    - type: nauc_recall_at_1_std
      value: -50.42816921722124
    - type: nauc_recall_at_20_diff1
      value: 76.31935254845239
    - type: nauc_recall_at_20_max
      value: 33.27135209054888
    - type: nauc_recall_at_20_std
      value: -99.45811267773247
    - type: nauc_recall_at_3_diff1
      value: 75.5835970617533
    - type: nauc_recall_at_3_max
      value: 27.405242767336773
    - type: nauc_recall_at_3_std
      value: -74.17660685878087
    - type: nauc_recall_at_5_diff1
      value: 74.59626833020839
    - type: nauc_recall_at_5_max
      value: 29.614350732835216
    - type: nauc_recall_at_5_std
      value: -82.79699422821245
    - type: ndcg_at_1
      value: 83.78
    - type: ndcg_at_10
      value: 90.361
    - type: ndcg_at_100
      value: 91.32400000000001
    - type: ndcg_at_1000
      value: 91.381
    - type: ndcg_at_20
      value: 90.931
    - type: ndcg_at_3
      value: 87.888
    - type: ndcg_at_5
      value: 89.319
    - type: precision_at_1
      value: 83.78
    - type: precision_at_10
      value: 13.709
    - type: precision_at_100
      value: 1.542
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_20
      value: 7.247000000000001
    - type: precision_at_3
      value: 38.58
    - type: precision_at_5
      value: 25.291999999999998
    - type: recall_at_1
      value: 72.68
    - type: recall_at_10
      value: 96.666
    - type: recall_at_100
      value: 99.764
    - type: recall_at_1000
      value: 99.996
    - type: recall_at_20
      value: 98.476
    - type: recall_at_3
      value: 89.39200000000001
    - type: recall_at_5
      value: 93.58800000000001
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB RedditClustering (default)
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
      split: test
      type: mteb/reddit-clustering
    metrics:
    - type: main_score
      value: 64.73482304388713
    - type: v_measure
      value: 64.73482304388713
    - type: v_measure_std
      value: 3.2012056543499723
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB RedditClusteringP2P (default)
      revision: 385e3cb46b4cfa89021f56c4380204149d0efe33
      split: test
      type: mteb/reddit-clustering-p2p
    metrics:
    - type: main_score
      value: 69.09527946391182
    - type: v_measure
      value: 69.09527946391182
    - type: v_measure_std
      value: 11.917819058371844
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB SCIDOCS (default)
      revision: f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88
      split: test
      type: mteb/scidocs
    metrics:
    - type: main_score
      value: 25.288
    - type: map_at_1
      value: 5.833
    - type: map_at_10
      value: 15.747
    - type: map_at_100
      value: 18.412
    - type: map_at_1000
      value: 18.794
    - type: map_at_20
      value: 17.081
    - type: map_at_3
      value: 11.097
    - type: map_at_5
      value: 13.455
    - type: mrr_at_1
      value: 28.799999999999997
    - type: mrr_at_10
      value: 41.14087301587302
    - type: mrr_at_100
      value: 42.21477191496872
    - type: mrr_at_1000
      value: 42.24957221015803
    - type: mrr_at_20
      value: 41.82780604543376
    - type: mrr_at_3
      value: 38.03333333333333
    - type: mrr_at_5
      value: 39.72833333333334
    - type: nauc_map_at_1000_diff1
      value: 17.11054147719995
    - type: nauc_map_at_1000_max
      value: 28.085457595449938
    - type: nauc_map_at_1000_std
      value: 10.179619109879317
    - type: nauc_map_at_100_diff1
      value: 17.067499049316947
    - type: nauc_map_at_100_max
      value: 28.067558750663054
    - type: nauc_map_at_100_std
      value: 10.095105057362373
    - type: nauc_map_at_10_diff1
      value: 17.27051021394208
    - type: nauc_map_at_10_max
      value: 26.681322874549746
    - type: nauc_map_at_10_std
      value: 5.926268838785555
    - type: nauc_map_at_1_diff1
      value: 24.33996740078769
    - type: nauc_map_at_1_max
      value: 22.553300613884193
    - type: nauc_map_at_1_std
      value: 1.0179814188998704
    - type: nauc_map_at_20_diff1
      value: 17.318508033275275
    - type: nauc_map_at_20_max
      value: 27.480260696622828
    - type: nauc_map_at_20_std
      value: 8.05317136384471
    - type: nauc_map_at_3_diff1
      value: 20.873591821674705
    - type: nauc_map_at_3_max
      value: 25.41326252731943
    - type: nauc_map_at_3_std
      value: 0.41336392743959216
    - type: nauc_map_at_5_diff1
      value: 18.763809198559635
    - type: nauc_map_at_5_max
      value: 25.898376905475512
    - type: nauc_map_at_5_std
      value: 2.988175157354391
    - type: nauc_mrr_at_1000_diff1
      value: 22.469264232988433
    - type: nauc_mrr_at_1000_max
      value: 26.644886550441445
    - type: nauc_mrr_at_1000_std
      value: 6.3513361738005925
    - type: nauc_mrr_at_100_diff1
      value: 22.461607961871795
    - type: nauc_mrr_at_100_max
      value: 26.663957731096943
    - type: nauc_mrr_at_100_std
      value: 6.381995245140103
    - type: nauc_mrr_at_10_diff1
      value: 22.621389127374748
    - type: nauc_mrr_at_10_max
      value: 26.857129444585325
    - type: nauc_mrr_at_10_std
      value: 6.582137920004729
    - type: nauc_mrr_at_1_diff1
      value: 24.53157041741088
    - type: nauc_mrr_at_1_max
      value: 23.05475623470265
    - type: nauc_mrr_at_1_std
      value: 1.179825141672729
    - type: nauc_mrr_at_20_diff1
      value: 22.412018526214926
    - type: nauc_mrr_at_20_max
      value: 26.65322991962089
    - type: nauc_mrr_at_20_std
      value: 6.296449643563243
    - type: nauc_mrr_at_3_diff1
      value: 22.437609855261943
    - type: nauc_mrr_at_3_max
      value: 25.681928857181525
    - type: nauc_mrr_at_3_std
      value: 4.553634098623652
    - type: nauc_mrr_at_5_diff1
      value: 22.023453611839226
    - type: nauc_mrr_at_5_max
      value: 26.40903357360535
    - type: nauc_mrr_at_5_std
      value: 5.89093223105796
    - type: nauc_ndcg_at_1000_diff1
      value: 17.399108097318532
    - type: nauc_ndcg_at_1000_max
      value: 29.79589138599995
    - type: nauc_ndcg_at_1000_std
      value: 16.697992042743145
    - type: nauc_ndcg_at_100_diff1
      value: 17.172628053007347
    - type: nauc_ndcg_at_100_max
      value: 30.431134527142017
    - type: nauc_ndcg_at_100_std
      value: 17.44560703748887
    - type: nauc_ndcg_at_10_diff1
      value: 18.063436302614008
    - type: nauc_ndcg_at_10_max
      value: 28.397194973415846
    - type: nauc_ndcg_at_10_std
      value: 9.268714839396178
    - type: nauc_ndcg_at_1_diff1
      value: 24.53157041741088
    - type: nauc_ndcg_at_1_max
      value: 23.05475623470265
    - type: nauc_ndcg_at_1_std
      value: 1.179825141672729
    - type: nauc_ndcg_at_20_diff1
      value: 17.741030215784328
    - type: nauc_ndcg_at_20_max
      value: 29.359849195331623
    - type: nauc_ndcg_at_20_std
      value: 12.204529054561519
    - type: nauc_ndcg_at_3_diff1
      value: 20.70556071477731
    - type: nauc_ndcg_at_3_max
      value: 26.033995467225562
    - type: nauc_ndcg_at_3_std
      value: 2.343922717548071
    - type: nauc_ndcg_at_5_diff1
      value: 18.891397898543147
    - type: nauc_ndcg_at_5_max
      value: 26.927407037356016
    - type: nauc_ndcg_at_5_std
      value: 5.953588591672794
    - type: nauc_precision_at_1000_diff1
      value: 3.8043849987759053
    - type: nauc_precision_at_1000_max
      value: 21.147369019911533
    - type: nauc_precision_at_1000_std
      value: 32.67737549532435
    - type: nauc_precision_at_100_diff1
      value: 8.27055390106616
    - type: nauc_precision_at_100_max
      value: 27.167823849615207
    - type: nauc_precision_at_100_std
      value: 31.02390080705758
    - type: nauc_precision_at_10_diff1
      value: 13.459563852304363
    - type: nauc_precision_at_10_max
      value: 28.480266596483673
    - type: nauc_precision_at_10_std
      value: 13.834835842251387
    - type: nauc_precision_at_1_diff1
      value: 24.53157041741088
    - type: nauc_precision_at_1_max
      value: 23.05475623470265
    - type: nauc_precision_at_1_std
      value: 1.179825141672729
    - type: nauc_precision_at_20_diff1
      value: 11.728307323617736
    - type: nauc_precision_at_20_max
      value: 28.24616729081262
    - type: nauc_precision_at_20_std
      value: 19.02176870002311
    - type: nauc_precision_at_3_diff1
      value: 18.758006603429163
    - type: nauc_precision_at_3_max
      value: 26.23015904698099
    - type: nauc_precision_at_3_std
      value: 2.2900065250415578
    - type: nauc_precision_at_5_diff1
      value: 15.234379429768971
    - type: nauc_precision_at_5_max
      value: 26.945809679548383
    - type: nauc_precision_at_5_std
      value: 8.501867732343884
    - type: nauc_recall_at_1000_diff1
      value: 3.8370674833381706
    - type: nauc_recall_at_1000_max
      value: 20.24705690708739
    - type: nauc_recall_at_1000_std
      value: 34.29336092406221
    - type: nauc_recall_at_100_diff1
      value: 8.202793314438631
    - type: nauc_recall_at_100_max
      value: 26.630338769496475
    - type: nauc_recall_at_100_std
      value: 31.212408261143633
    - type: nauc_recall_at_10_diff1
      value: 13.186227511395327
    - type: nauc_recall_at_10_max
      value: 27.806226574014616
    - type: nauc_recall_at_10_std
      value: 13.618825668274305
    - type: nauc_recall_at_1_diff1
      value: 24.33996740078769
    - type: nauc_recall_at_1_max
      value: 22.553300613884193
    - type: nauc_recall_at_1_std
      value: 1.0179814188998704
    - type: nauc_recall_at_20_diff1
      value: 11.456283962336089
    - type: nauc_recall_at_20_max
      value: 27.804662740333143
    - type: nauc_recall_at_20_std
      value: 18.998766577013964
    - type: nauc_recall_at_3_diff1
      value: 18.568901463738698
    - type: nauc_recall_at_3_max
      value: 25.567582570409492
    - type: nauc_recall_at_3_std
      value: 2.040055880348112
    - type: nauc_recall_at_5_diff1
      value: 15.021259180944146
    - type: nauc_recall_at_5_max
      value: 26.29579526626801
    - type: nauc_recall_at_5_std
      value: 8.276410907808483
    - type: ndcg_at_1
      value: 28.799999999999997
    - type: ndcg_at_10
      value: 25.288
    - type: ndcg_at_100
      value: 34.802
    - type: ndcg_at_1000
      value: 40.611999999999995
    - type: ndcg_at_20
      value: 28.786
    - type: ndcg_at_3
      value: 24.329
    - type: ndcg_at_5
      value: 21.204
    - type: precision_at_1
      value: 28.799999999999997
    - type: precision_at_10
      value: 13.059999999999999
    - type: precision_at_100
      value: 2.67
    - type: precision_at_1000
      value: 0.406
    - type: precision_at_20
      value: 8.555
    - type: precision_at_3
      value: 23.033
    - type: precision_at_5
      value: 18.759999999999998
    - type: recall_at_1
      value: 5.833
    - type: recall_at_10
      value: 26.47
    - type: recall_at_100
      value: 54.21
    - type: recall_at_1000
      value: 82.312
    - type: recall_at_20
      value: 34.675
    - type: recall_at_3
      value: 14.008000000000001
    - type: recall_at_5
      value: 19.006999999999998
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB SICK-R (default)
      revision: 20a6d6f312dd54037fe07a32d58e5e168867909d
      split: test
      type: mteb/sickr-sts
    metrics:
    - type: cosine_pearson
      value: 80.56086069694913
    - type: cosine_spearman
      value: 79.93983971081498
    - type: euclidean_pearson
      value: 72.90001682627418
    - type: euclidean_spearman
      value: 72.98053904390399
    - type: main_score
      value: 79.93983971081498
    - type: manhattan_pearson
      value: 72.86370864289793
    - type: manhattan_spearman
      value: 72.95043309386041
    - type: pearson
      value: 80.56086069694913
    - type: spearman
      value: 79.93983971081498
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS12 (default)
      revision: a0d554a64d88156834ff5ae9920b964011b16384
      split: test
      type: mteb/sts12-sts
    metrics:
    - type: cosine_pearson
      value: 85.46442665468061
    - type: cosine_spearman
      value: 79.01640858560742
    - type: euclidean_pearson
      value: 72.55686689261881
    - type: euclidean_spearman
      value: 66.56332608235445
    - type: main_score
      value: 79.01640858560742
    - type: manhattan_pearson
      value: 72.53251656292508
    - type: manhattan_spearman
      value: 66.57471256950211
    - type: pearson
      value: 85.46442665468061
    - type: spearman
      value: 79.01640858560742
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS13 (default)
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
      split: test
      type: mteb/sts13-sts
    metrics:
    - type: cosine_pearson
      value: 83.06969715132047
    - type: cosine_spearman
      value: 83.60535698131366
    - type: euclidean_pearson
      value: 77.86493835600183
    - type: euclidean_spearman
      value: 79.05632834319427
    - type: main_score
      value: 83.60535698131366
    - type: manhattan_pearson
      value: 77.89229232585566
    - type: manhattan_spearman
      value: 79.08309882862123
    - type: pearson
      value: 83.06969715132047
    - type: spearman
      value: 83.60535698131366
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS14 (default)
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
      split: test
      type: mteb/sts14-sts
    metrics:
    - type: cosine_pearson
      value: 81.80825964650951
    - type: cosine_spearman
      value: 82.1588910393309
    - type: euclidean_pearson
      value: 75.03149873671988
    - type: euclidean_spearman
      value: 75.61594820888504
    - type: main_score
      value: 82.1588910393309
    - type: manhattan_pearson
      value: 75.04795190923636
    - type: manhattan_spearman
      value: 75.64525963935924
    - type: pearson
      value: 81.80825964650951
    - type: spearman
      value: 82.1588910393309
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS15 (default)
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
      split: test
      type: mteb/sts15-sts
    metrics:
    - type: cosine_pearson
      value: 87.44980743291514
    - type: cosine_spearman
      value: 87.58883022167446
    - type: euclidean_pearson
      value: 72.14550902364643
    - type: euclidean_spearman
      value: 72.55754297396044
    - type: main_score
      value: 87.58883022167446
    - type: manhattan_pearson
      value: 72.13178863025279
    - type: manhattan_spearman
      value: 72.54754623550077
    - type: pearson
      value: 87.44980743291514
    - type: spearman
      value: 87.58883022167446
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS16 (default)
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
      split: test
      type: mteb/sts16-sts
    metrics:
    - type: cosine_pearson
      value: 85.28696815842446
    - type: cosine_spearman
      value: 86.14258330738964
    - type: euclidean_pearson
      value: 72.58270502711088
    - type: euclidean_spearman
      value: 73.90360832137756
    - type: main_score
      value: 86.14258330738964
    - type: manhattan_pearson
      value: 72.56781719390185
    - type: manhattan_spearman
      value: 73.82667513596357
    - type: pearson
      value: 85.28696815842446
    - type: spearman
      value: 86.14258330738964
    task:
      type: STS
  - dataset:
      config: en-en
      name: MTEB STS17 (en-en)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 87.32736181779451
    - type: cosine_spearman
      value: 86.19027879386091
    - type: euclidean_pearson
      value: 72.38065753765687
    - type: euclidean_spearman
      value: 69.00098758155796
    - type: main_score
      value: 86.19027879386091
    - type: manhattan_pearson
      value: 72.28337749494482
    - type: manhattan_spearman
      value: 68.90623383171824
    - type: pearson
      value: 87.32736181779451
    - type: spearman
      value: 86.19027879386091
    task:
      type: STS
  - dataset:
      config: en
      name: MTEB STS22 (en)
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 68.08171402030908
    - type: cosine_spearman
      value: 66.39158241439456
    - type: euclidean_pearson
      value: 49.253669417684364
    - type: euclidean_spearman
      value: 59.37216217028178
    - type: main_score
      value: 66.39158241439456
    - type: manhattan_pearson
      value: 49.569435153138606
    - type: manhattan_spearman
      value: 59.69933577872874
    - type: pearson
      value: 68.08171402030908
    - type: spearman
      value: 66.39158241439456
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STSBenchmark (default)
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
      split: test
      type: mteb/stsbenchmark-sts
    metrics:
    - type: cosine_pearson
      value: 85.32686607427509
    - type: cosine_spearman
      value: 86.12364023358859
    - type: euclidean_pearson
      value: 72.60010510555432
    - type: euclidean_spearman
      value: 73.37115371181623
    - type: main_score
      value: 86.12364023358859
    - type: manhattan_pearson
      value: 72.60389334024258
    - type: manhattan_spearman
      value: 73.37360729715007
    - type: pearson
      value: 85.32686607427509
    - type: spearman
      value: 86.12364023358859
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB SciDocsRR (default)
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
      split: test
      type: mteb/scidocs-reranking
    metrics:
    - type: main_score
      value: 86.16941293184462
    - type: map
      value: 86.16941293184462
    - type: mrr
      value: 96.292615336733
    - type: nAUC_map_diff1
      value: 3.470227813316292
    - type: nAUC_map_max
      value: 54.91440902009646
    - type: nAUC_map_std
      value: 69.88258003404158
    - type: nAUC_mrr_diff1
      value: 51.65993469951814
    - type: nAUC_mrr_max
      value: 89.8476301315306
    - type: nAUC_mrr_std
      value: 89.63560910860683
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB SciFact (default)
      revision: 0228b52cf27578f30900b9e5271d331663a030d7
      split: test
      type: mteb/scifact
    metrics:
    - type: main_score
      value: 76.506
    - type: map_at_1
      value: 63.161
    - type: map_at_10
      value: 72.329
    - type: map_at_100
      value: 72.76599999999999
    - type: map_at_1000
      value: 72.785
    - type: map_at_20
      value: 72.664
    - type: map_at_3
      value: 69.807
    - type: map_at_5
      value: 71.072
    - type: mrr_at_1
      value: 66.33333333333333
    - type: mrr_at_10
      value: 73.49603174603175
    - type: mrr_at_100
      value: 73.83699546695885
    - type: mrr_at_1000
      value: 73.85615365073951
    - type: mrr_at_20
      value: 73.73908372329426
    - type: mrr_at_3
      value: 71.88888888888889
    - type: mrr_at_5
      value: 72.57222222222222
    - type: nauc_map_at_1000_diff1
      value: 68.69164426077057
    - type: nauc_map_at_1000_max
      value: 49.68097438309853
    - type: nauc_map_at_1000_std
      value: 3.568465767044567
    - type: nauc_map_at_100_diff1
      value: 68.69528225462155
    - type: nauc_map_at_100_max
      value: 49.68997783637044
    - type: nauc_map_at_100_std
      value: 3.5405870639903734
    - type: nauc_map_at_10_diff1
      value: 68.60029590321635
    - type: nauc_map_at_10_max
      value: 49.9660074190514
    - type: nauc_map_at_10_std
      value: 3.648393629556409
    - type: nauc_map_at_1_diff1
      value: 72.04288973445112
    - type: nauc_map_at_1_max
      value: 44.44343257926281
    - type: nauc_map_at_1_std
      value: -1.1896886029045166
    - type: nauc_map_at_20_diff1
      value: 68.53206363790187
    - type: nauc_map_at_20_max
      value: 49.6543815607468
    - type: nauc_map_at_20_std
      value: 3.491882748903082
    - type: nauc_map_at_3_diff1
      value: 68.51164002388202
    - type: nauc_map_at_3_max
      value: 46.088782010642625
    - type: nauc_map_at_3_std
      value: 0.6715722116439257
    - type: nauc_map_at_5_diff1
      value: 68.92061622098807
    - type: nauc_map_at_5_max
      value: 49.494693431176856
    - type: nauc_map_at_5_std
      value: 3.2644304939841375
    - type: nauc_mrr_at_1000_diff1
      value: 69.22693174497351
    - type: nauc_mrr_at_1000_max
      value: 51.73749232061987
    - type: nauc_mrr_at_1000_std
      value: 6.971620031265618
    - type: nauc_mrr_at_100_diff1
      value: 69.2302127326908
    - type: nauc_mrr_at_100_max
      value: 51.744990295219765
    - type: nauc_mrr_at_100_std
      value: 6.94009832111095
    - type: nauc_mrr_at_10_diff1
      value: 69.044420978255
    - type: nauc_mrr_at_10_max
      value: 51.94860197061985
    - type: nauc_mrr_at_10_std
      value: 7.298432524786719
    - type: nauc_mrr_at_1_diff1
      value: 72.08706643576656
    - type: nauc_mrr_at_1_max
      value: 50.28314967119001
    - type: nauc_mrr_at_1_std
      value: 5.627912933564255
    - type: nauc_mrr_at_20_diff1
      value: 69.05595939713633
    - type: nauc_mrr_at_20_max
      value: 51.69581294232438
    - type: nauc_mrr_at_20_std
      value: 6.86902093014737
    - type: nauc_mrr_at_3_diff1
      value: 69.23425861932469
    - type: nauc_mrr_at_3_max
      value: 51.189149054992896
    - type: nauc_mrr_at_3_std
      value: 7.148160590220123
    - type: nauc_mrr_at_5_diff1
      value: 69.39770871231859
    - type: nauc_mrr_at_5_max
      value: 52.44031078141917
    - type: nauc_mrr_at_5_std
      value: 7.9520400234720094
    - type: nauc_ndcg_at_1000_diff1
      value: 68.50681430607833
    - type: nauc_ndcg_at_1000_max
      value: 51.395626005233794
    - type: nauc_ndcg_at_1000_std
      value: 5.3215128458056
    - type: nauc_ndcg_at_100_diff1
      value: 68.5973705398806
    - type: nauc_ndcg_at_100_max
      value: 51.70299248501594
    - type: nauc_ndcg_at_100_std
      value: 5.021306609806647
    - type: nauc_ndcg_at_10_diff1
      value: 67.64313041478901
    - type: nauc_ndcg_at_10_max
      value: 52.65287998706653
    - type: nauc_ndcg_at_10_std
      value: 5.624893544542878
    - type: nauc_ndcg_at_1_diff1
      value: 72.08706643576656
    - type: nauc_ndcg_at_1_max
      value: 50.28314967119001
    - type: nauc_ndcg_at_1_std
      value: 5.627912933564255
    - type: nauc_ndcg_at_20_diff1
      value: 67.40463404794681
    - type: nauc_ndcg_at_20_max
      value: 51.589370586322744
    - type: nauc_ndcg_at_20_std
      value: 4.716470449272336
    - type: nauc_ndcg_at_3_diff1
      value: 67.77455934283292
    - type: nauc_ndcg_at_3_max
      value: 48.56277444161093
    - type: nauc_ndcg_at_3_std
      value: 3.4981418225876033
    - type: nauc_ndcg_at_5_diff1
      value: 68.62063480904568
    - type: nauc_ndcg_at_5_max
      value: 52.51840344159725
    - type: nauc_ndcg_at_5_std
      value: 5.869056344601484
    - type: nauc_precision_at_1000_diff1
      value: -26.953004065963587
    - type: nauc_precision_at_1000_max
      value: 16.670450558296018
    - type: nauc_precision_at_1000_std
      value: 37.79075767270023
    - type: nauc_precision_at_100_diff1
      value: -12.137474642955286
    - type: nauc_precision_at_100_max
      value: 23.66287260971483
    - type: nauc_precision_at_100_std
      value: 32.05656594000832
    - type: nauc_precision_at_10_diff1
      value: 4.693636678130231
    - type: nauc_precision_at_10_max
      value: 37.1192079933253
    - type: nauc_precision_at_10_std
      value: 29.361429356341944
    - type: nauc_precision_at_1_diff1
      value: 72.08706643576656
    - type: nauc_precision_at_1_max
      value: 50.28314967119001
    - type: nauc_precision_at_1_std
      value: 5.627912933564255
    - type: nauc_precision_at_20_diff1
      value: -5.581166313936688
    - type: nauc_precision_at_20_max
      value: 28.095991450663178
    - type: nauc_precision_at_20_std
      value: 28.70785575089581
    - type: nauc_precision_at_3_diff1
      value: 38.039786794046506
    - type: nauc_precision_at_3_max
      value: 42.80662919057468
    - type: nauc_precision_at_3_std
      value: 15.791221273992933
    - type: nauc_precision_at_5_diff1
      value: 22.865160964940728
    - type: nauc_precision_at_5_max
      value: 45.69237145525087
    - type: nauc_precision_at_5_std
      value: 26.540864172255198
    - type: nauc_recall_at_1000_diff1
      value: .nan
    - type: nauc_recall_at_1000_max
      value: .nan
    - type: nauc_recall_at_1000_std
      value: .nan
    - type: nauc_recall_at_100_diff1
      value: 71.64254590725189
    - type: nauc_recall_at_100_max
      value: 63.20417055711188
    - type: nauc_recall_at_100_std
      value: -0.36959228135702915
    - type: nauc_recall_at_10_diff1
      value: 60.08308613307207
    - type: nauc_recall_at_10_max
      value: 62.08842834324362
    - type: nauc_recall_at_10_std
      value: 7.674393632300139
    - type: nauc_recall_at_1_diff1
      value: 72.04288973445112
    - type: nauc_recall_at_1_max
      value: 44.44343257926281
    - type: nauc_recall_at_1_std
      value: -1.1896886029045166
    - type: nauc_recall_at_20_diff1
      value: 55.127441047340945
    - type: nauc_recall_at_20_max
      value: 57.231261298845716
    - type: nauc_recall_at_20_std
      value: -1.0181377515545467
    - type: nauc_recall_at_3_diff1
      value: 64.26346387076399
    - type: nauc_recall_at_3_max
      value: 47.967778804933126
    - type: nauc_recall_at_3_std
      value: 2.000342237324162
    - type: nauc_recall_at_5_diff1
      value: 66.2721108117289
    - type: nauc_recall_at_5_max
      value: 59.00393323656199
    - type: nauc_recall_at_5_std
      value: 8.922186068023741
    - type: ndcg_at_1
      value: 66.333
    - type: ndcg_at_10
      value: 76.506
    - type: ndcg_at_100
      value: 78.289
    - type: ndcg_at_1000
      value: 78.803
    - type: ndcg_at_20
      value: 77.568
    - type: ndcg_at_3
      value: 72.40100000000001
    - type: ndcg_at_5
      value: 73.991
    - type: precision_at_1
      value: 66.333
    - type: precision_at_10
      value: 10.033
    - type: precision_at_100
      value: 1.09
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_20
      value: 5.25
    - type: precision_at_3
      value: 27.889000000000003
    - type: precision_at_5
      value: 18.133
    - type: recall_at_1
      value: 63.161
    - type: recall_at_10
      value: 88.167
    - type: recall_at_100
      value: 96.0
    - type: recall_at_1000
      value: 100.0
    - type: recall_at_20
      value: 92.167
    - type: recall_at_3
      value: 77.089
    - type: recall_at_5
      value: 80.95599999999999
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB SprintDuplicateQuestions (default)
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
      split: test
      type: mteb/sprintduplicatequestions-pairclassification
    metrics:
    - type: cosine_accuracy
      value: 99.84455445544555
    - type: cosine_accuracy_threshold
      value: 76.59906148910522
    - type: cosine_ap
      value: 96.575017942212
    - type: cosine_f1
      value: 91.92734611503532
    - type: cosine_f1_threshold
      value: 74.90451335906982
    - type: cosine_precision
      value: 92.76985743380855
    - type: cosine_recall
      value: 91.10000000000001
    - type: dot_accuracy
      value: 99.23960396039604
    - type: dot_accuracy_threshold
      value: 31528.216552734375
    - type: dot_ap
      value: 57.86944962370526
    - type: dot_f1
      value: 56.55655655655656
    - type: dot_f1_threshold
      value: 26655.227661132812
    - type: dot_precision
      value: 56.613226452905806
    - type: dot_recall
      value: 56.49999999999999
    - type: euclidean_accuracy
      value: 99.64752475247525
    - type: euclidean_accuracy_threshold
      value: 1188.8517379760742
    - type: euclidean_ap
      value: 86.69227976710795
    - type: euclidean_f1
      value: 81.00734522560334
    - type: euclidean_f1_threshold
      value: 1237.8013610839844
    - type: euclidean_precision
      value: 85.20971302428256
    - type: euclidean_recall
      value: 77.2
    - type: main_score
      value: 96.575017942212
    - type: manhattan_accuracy
      value: 99.64653465346535
    - type: manhattan_accuracy_threshold
      value: 41083.306884765625
    - type: manhattan_ap
      value: 86.63228824679388
    - type: manhattan_f1
      value: 80.8645229309436
    - type: manhattan_f1_threshold
      value: 44612.701416015625
    - type: manhattan_precision
      value: 85.5072463768116
    - type: manhattan_recall
      value: 76.7
    - type: max_accuracy
      value: 99.84455445544555
    - type: max_ap
      value: 96.575017942212
    - type: max_f1
      value: 91.92734611503532
    - type: max_precision
      value: 92.76985743380855
    - type: max_recall
      value: 91.10000000000001
    - type: similarity_accuracy
      value: 99.84455445544555
    - type: similarity_accuracy_threshold
      value: 76.59906148910522
    - type: similarity_ap
      value: 96.575017942212
    - type: similarity_f1
      value: 91.92734611503532
    - type: similarity_f1_threshold
      value: 74.90451335906982
    - type: similarity_precision
      value: 92.76985743380855
    - type: similarity_recall
      value: 91.10000000000001
    task:
      type: PairClassification
  - dataset:
      config: default
      name: MTEB StackExchangeClustering (default)
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
      split: test
      type: mteb/stackexchange-clustering
    metrics:
    - type: main_score
      value: 75.52180723638888
    - type: v_measure
      value: 75.52180723638888
    - type: v_measure_std
      value: 3.2684011739397794
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB StackExchangeClusteringP2P (default)
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
      split: test
      type: mteb/stackexchange-clustering-p2p
    metrics:
    - type: main_score
      value: 42.96437441541306
    - type: v_measure
      value: 42.96437441541306
    - type: v_measure_std
      value: 1.442861756284192
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB StackOverflowDupQuestions (default)
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
      split: test
      type: mteb/stackoverflowdupquestions-reranking
    metrics:
    - type: main_score
      value: 56.39462784328071
    - type: map
      value: 56.39462784328071
    - type: mrr
      value: 57.43936210847976
    - type: nAUC_map_diff1
      value: 40.50286714673592
    - type: nAUC_map_max
      value: 10.224449587081825
    - type: nAUC_map_std
      value: 6.368408763477458
    - type: nAUC_mrr_diff1
      value: 40.567497774555854
    - type: nAUC_mrr_max
      value: 11.732191328482237
    - type: nAUC_mrr_std
      value: 6.808587430834912
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB SummEval (default)
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
      split: test
      type: mteb/summeval
    metrics:
    - type: cosine_pearson
      value: 32.039484755290886
    - type: cosine_spearman
      value: 32.509024568368986
    - type: dot_pearson
      value: 22.6607759947711
    - type: dot_spearman
      value: 23.72878403008937
    - type: main_score
      value: 32.509024568368986
    - type: pearson
      value: 32.039484755290886
    - type: spearman
      value: 32.509024568368986
    task:
      type: Summarization
  - dataset:
      config: default
      name: MTEB TRECCOVID (default)
      revision: bb9466bac8153a0349341eb1b22e06409e78ef4e
      split: test
      type: mteb/trec-covid
    metrics:
    - type: main_score
      value: 83.11
    - type: map_at_1
      value: 0.251
    - type: map_at_10
      value: 2.119
    - type: map_at_100
      value: 12.767999999999999
    - type: map_at_1000
      value: 31.837
    - type: map_at_20
      value: 3.8120000000000003
    - type: map_at_3
      value: 0.719
    - type: map_at_5
      value: 1.142
    - type: mrr_at_1
      value: 94.0
    - type: mrr_at_10
      value: 97.0
    - type: mrr_at_100
      value: 97.0
    - type: mrr_at_1000
      value: 97.0
    - type: mrr_at_20
      value: 97.0
    - type: mrr_at_3
      value: 97.0
    - type: mrr_at_5
      value: 97.0
    - type: nauc_map_at_1000_diff1
      value: -10.833971330653577
    - type: nauc_map_at_1000_max
      value: 49.43623513591739
    - type: nauc_map_at_1000_std
      value: 74.92725410728191
    - type: nauc_map_at_100_diff1
      value: 3.0169264163343
    - type: nauc_map_at_100_max
      value: 39.64656463781187
    - type: nauc_map_at_100_std
      value: 52.685437762465625
    - type: nauc_map_at_10_diff1
      value: -3.663621519753861
    - type: nauc_map_at_10_max
      value: 22.278678114406265
    - type: nauc_map_at_10_std
      value: 11.871036844728627
    - type: nauc_map_at_1_diff1
      value: 12.092809371028176
    - type: nauc_map_at_1_max
      value: 6.962525273224316
    - type: nauc_map_at_1_std
      value: -3.482535899864613
    - type: nauc_map_at_20_diff1
      value: -4.47558912782466
    - type: nauc_map_at_20_max
      value: 27.05435806642965
    - type: nauc_map_at_20_std
      value: 22.50431991499924
    - type: nauc_map_at_3_diff1
      value: 1.6762978546843328
    - type: nauc_map_at_3_max
      value: 16.073549778056453
    - type: nauc_map_at_3_std
      value: 1.5936026368762652
    - type: nauc_map_at_5_diff1
      value: -5.920042861160988
    - type: nauc_map_at_5_max
      value: 17.49970473965879
    - type: nauc_map_at_5_std
      value: 6.683184754006298
    - type: nauc_mrr_at_1000_diff1
      value: 17.22689075630276
    - type: nauc_mrr_at_1000_max
      value: 75.879240585123
    - type: nauc_mrr_at_1000_std
      value: 74.2452536570186
    - type: nauc_mrr_at_100_diff1
      value: 17.22689075630276
    - type: nauc_mrr_at_100_max
      value: 75.879240585123
    - type: nauc_mrr_at_100_std
      value: 74.2452536570186
    - type: nauc_mrr_at_10_diff1
      value: 17.22689075630276
    - type: nauc_mrr_at_10_max
      value: 75.879240585123
    - type: nauc_mrr_at_10_std
      value: 74.2452536570186
    - type: nauc_mrr_at_1_diff1
      value: 17.226890756302502
    - type: nauc_mrr_at_1_max
      value: 75.87924058512304
    - type: nauc_mrr_at_1_std
      value: 74.24525365701842
    - type: nauc_mrr_at_20_diff1
      value: 17.22689075630276
    - type: nauc_mrr_at_20_max
      value: 75.879240585123
    - type: nauc_mrr_at_20_std
      value: 74.2452536570186
    - type: nauc_mrr_at_3_diff1
      value: 17.22689075630276
    - type: nauc_mrr_at_3_max
      value: 75.879240585123
    - type: nauc_mrr_at_3_std
      value: 74.2452536570186
    - type: nauc_mrr_at_5_diff1
      value: 17.22689075630276
    - type: nauc_mrr_at_5_max
      value: 75.879240585123
    - type: nauc_mrr_at_5_std
      value: 74.2452536570186
    - type: nauc_ndcg_at_1000_diff1
      value: -10.239823772202964
    - type: nauc_ndcg_at_1000_max
      value: 47.40567642788709
    - type: nauc_ndcg_at_1000_std
      value: 73.16002291789529
    - type: nauc_ndcg_at_100_diff1
      value: -1.784781407477811
    - type: nauc_ndcg_at_100_max
      value: 47.098382192293585
    - type: nauc_ndcg_at_100_std
      value: 75.8409480683384
    - type: nauc_ndcg_at_10_diff1
      value: -1.479834920042459
    - type: nauc_ndcg_at_10_max
      value: 43.755901787149554
    - type: nauc_ndcg_at_10_std
      value: 56.777378887073226
    - type: nauc_ndcg_at_1_diff1
      value: 23.651970052181902
    - type: nauc_ndcg_at_1_max
      value: 22.0108901157074
    - type: nauc_ndcg_at_1_std
      value: 36.31551085230281
    - type: nauc_ndcg_at_20_diff1
      value: -2.8462288798935633
    - type: nauc_ndcg_at_20_max
      value: 44.14781928337475
    - type: nauc_ndcg_at_20_std
      value: 64.97168491870123
    - type: nauc_ndcg_at_3_diff1
      value: 9.234801311530015
    - type: nauc_ndcg_at_3_max
      value: 37.473059801435255
    - type: nauc_ndcg_at_3_std
      value: 47.91443188012391
    - type: nauc_ndcg_at_5_diff1
      value: -4.058832252029437
    - type: nauc_ndcg_at_5_max
      value: 40.76028587335579
    - type: nauc_ndcg_at_5_std
      value: 54.981195955052854
    - type: nauc_precision_at_1000_diff1
      value: -18.070822675299915
    - type: nauc_precision_at_1000_max
      value: 34.26674387351181
    - type: nauc_precision_at_1000_std
      value: 46.365234391730795
    - type: nauc_precision_at_100_diff1
      value: -8.133355562733763
    - type: nauc_precision_at_100_max
      value: 50.88466067167518
    - type: nauc_precision_at_100_std
      value: 78.66168589162467
    - type: nauc_precision_at_10_diff1
      value: -18.45823463810709
    - type: nauc_precision_at_10_max
      value: 62.019004783454044
    - type: nauc_precision_at_10_std
      value: 66.78310399174201
    - type: nauc_precision_at_1_diff1
      value: 17.226890756302502
    - type: nauc_precision_at_1_max
      value: 75.87924058512304
    - type: nauc_precision_at_1_std
      value: 74.24525365701842
    - type: nauc_precision_at_20_diff1
      value: -15.627388961954306
    - type: nauc_precision_at_20_max
      value: 52.52691035437249
    - type: nauc_precision_at_20_std
      value: 70.72015767601786
    - type: nauc_precision_at_3_diff1
      value: -21.64967060734557
    - type: nauc_precision_at_3_max
      value: 84.4328129230213
    - type: nauc_precision_at_3_std
      value: 74.86688926992142
    - type: nauc_precision_at_5_diff1
      value: -33.6652946491316
    - type: nauc_precision_at_5_max
      value: 68.42084128099583
    - type: nauc_precision_at_5_std
      value: 72.99668707961033
    - type: nauc_recall_at_1000_diff1
      value: -11.177600103068249
    - type: nauc_recall_at_1000_max
      value: 42.170800191921366
    - type: nauc_recall_at_1000_std
      value: 61.42365012659931
    - type: nauc_recall_at_100_diff1
      value: 4.398273070866929
    - type: nauc_recall_at_100_max
      value: 31.392234960708
    - type: nauc_recall_at_100_std
      value: 40.24869663309948
    - type: nauc_recall_at_10_diff1
      value: -0.6449178416027819
    - type: nauc_recall_at_10_max
      value: 16.112165737469038
    - type: nauc_recall_at_10_std
      value: 3.434161242800518
    - type: nauc_recall_at_1_diff1
      value: 12.092809371028176
    - type: nauc_recall_at_1_max
      value: 6.962525273224316
    - type: nauc_recall_at_1_std
      value: -3.482535899864613
    - type: nauc_recall_at_20_diff1
      value: -0.5074564742224971
    - type: nauc_recall_at_20_max
      value: 19.850178419364216
    - type: nauc_recall_at_20_std
      value: 12.821203965364694
    - type: nauc_recall_at_3_diff1
      value: -0.39856898399862506
    - type: nauc_recall_at_3_max
      value: 12.772840403439817
    - type: nauc_recall_at_3_std
      value: -2.255110094980606
    - type: nauc_recall_at_5_diff1
      value: -5.610440085797961
    - type: nauc_recall_at_5_max
      value: 14.54746682352851
    - type: nauc_recall_at_5_std
      value: 2.644494349141052
    - type: ndcg_at_1
      value: 87.0
    - type: ndcg_at_10
      value: 83.11
    - type: ndcg_at_100
      value: 63.257
    - type: ndcg_at_1000
      value: 57.402
    - type: ndcg_at_20
      value: 78.127
    - type: ndcg_at_3
      value: 86.827
    - type: ndcg_at_5
      value: 85.283
    - type: precision_at_1
      value: 94.0
    - type: precision_at_10
      value: 87.6
    - type: precision_at_100
      value: 64.8
    - type: precision_at_1000
      value: 25.369999999999997
    - type: precision_at_20
      value: 81.8
    - type: precision_at_3
      value: 92.667
    - type: precision_at_5
      value: 90.0
    - type: recall_at_1
      value: 0.251
    - type: recall_at_10
      value: 2.266
    - type: recall_at_100
      value: 15.692999999999998
    - type: recall_at_1000
      value: 54.236
    - type: recall_at_20
      value: 4.182
    - type: recall_at_3
      value: 0.736
    - type: recall_at_5
      value: 1.174
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB Touche2020 (default)
      revision: a34f9a33db75fa0cbb21bb5cfc3dae8dc8bec93f
      split: test
      type: mteb/touche2020
    metrics:
    - type: main_score
      value: 27.724
    - type: map_at_1
      value: 3.4000000000000004
    - type: map_at_10
      value: 10.850999999999999
    - type: map_at_100
      value: 16.81
    - type: map_at_1000
      value: 18.262
    - type: map_at_20
      value: 13.503000000000002
    - type: map_at_3
      value: 6.288
    - type: map_at_5
      value: 8.021
    - type: mrr_at_1
      value: 44.89795918367347
    - type: mrr_at_10
      value: 55.205701328150305
    - type: mrr_at_100
      value: 56.040476796779316
    - type: mrr_at_1000
      value: 56.040476796779316
    - type: mrr_at_20
      value: 55.88940338039977
    - type: mrr_at_3
      value: 51.02040816326531
    - type: mrr_at_5
      value: 53.673469387755105
    - type: nauc_map_at_1000_diff1
      value: 42.566374536947755
    - type: nauc_map_at_1000_max
      value: -4.81782877305709
    - type: nauc_map_at_1000_std
      value: -1.712403720465198
    - type: nauc_map_at_100_diff1
      value: 44.36229920820552
    - type: nauc_map_at_100_max
      value: -5.51280916705942
    - type: nauc_map_at_100_std
      value: -5.446902679321525
    - type: nauc_map_at_10_diff1
      value: 55.031705973017885
    - type: nauc_map_at_10_max
      value: -4.3888293115180925
    - type: nauc_map_at_10_std
      value: -19.49554090301015
    - type: nauc_map_at_1_diff1
      value: 53.051363797042306
    - type: nauc_map_at_1_max
      value: -17.088092365200094
    - type: nauc_map_at_1_std
      value: -28.724504958117336
    - type: nauc_map_at_20_diff1
      value: 48.94420782354818
    - type: nauc_map_at_20_max
      value: -4.933712169290876
    - type: nauc_map_at_20_std
      value: -17.969225009817684
    - type: nauc_map_at_3_diff1
      value: 58.94774050204772
    - type: nauc_map_at_3_max
      value: -9.660516112672365
    - type: nauc_map_at_3_std
      value: -23.71045398045578
    - type: nauc_map_at_5_diff1
      value: 55.41114928066834
    - type: nauc_map_at_5_max
      value: -9.2564274429595
    - type: nauc_map_at_5_std
      value: -20.165025277125686
    - type: nauc_mrr_at_1000_diff1
      value: 31.86418353273936
    - type: nauc_mrr_at_1000_max
      value: -19.49903198031971
    - type: nauc_mrr_at_1000_std
      value: -16.146473721500794
    - type: nauc_mrr_at_100_diff1
      value: 31.86418353273936
    - type: nauc_mrr_at_100_max
      value: -19.49903198031971
    - type: nauc_mrr_at_100_std
      value: -16.146473721500794
    - type: nauc_mrr_at_10_diff1
      value: 31.93254247642881
    - type: nauc_mrr_at_10_max
      value: -19.867234585871664
    - type: nauc_mrr_at_10_std
      value: -14.877194192676205
    - type: nauc_mrr_at_1_diff1
      value: 33.14382151181415
    - type: nauc_mrr_at_1_max
      value: -25.77860726473143
    - type: nauc_mrr_at_1_std
      value: -24.934811974694586
    - type: nauc_mrr_at_20_diff1
      value: 31.820593975417317
    - type: nauc_mrr_at_20_max
      value: -19.849513390376945
    - type: nauc_mrr_at_20_std
      value: -16.24283715295157
    - type: nauc_mrr_at_3_diff1
      value: 29.596075415570795
    - type: nauc_mrr_at_3_max
      value: -17.997211789054344
    - type: nauc_mrr_at_3_std
      value: -20.769151268009523
    - type: nauc_mrr_at_5_diff1
      value: 31.056759708616788
    - type: nauc_mrr_at_5_max
      value: -17.48028966054467
    - type: nauc_mrr_at_5_std
      value: -13.800114719840199
    - type: nauc_ndcg_at_1000_diff1
      value: 25.048359973930868
    - type: nauc_ndcg_at_1000_max
      value: -0.9392048157007669
    - type: nauc_ndcg_at_1000_std
      value: 22.55123313576986
    - type: nauc_ndcg_at_100_diff1
      value: 32.64095147258336
    - type: nauc_ndcg_at_100_max
      value: -8.736843143324768
    - type: nauc_ndcg_at_100_std
      value: 14.168450218186946
    - type: nauc_ndcg_at_10_diff1
      value: 36.84108580887157
    - type: nauc_ndcg_at_10_max
      value: -6.365231331962414
    - type: nauc_ndcg_at_10_std
      value: -10.067797854126992
    - type: nauc_ndcg_at_1_diff1
      value: 27.60620630482724
    - type: nauc_ndcg_at_1_max
      value: -23.204817359303085
    - type: nauc_ndcg_at_1_std
      value: -23.185301124570653
    - type: nauc_ndcg_at_20_diff1
      value: 36.818771970409124
    - type: nauc_ndcg_at_20_max
      value: -10.60023482106758
    - type: nauc_ndcg_at_20_std
      value: -12.150746008034732
    - type: nauc_ndcg_at_3_diff1
      value: 30.11461816451728
    - type: nauc_ndcg_at_3_max
      value: -7.828745282289232
    - type: nauc_ndcg_at_3_std
      value: -16.47751283110174
    - type: nauc_ndcg_at_5_diff1
      value: 30.75762492455489
    - type: nauc_ndcg_at_5_max
      value: -4.800755841165072
    - type: nauc_ndcg_at_5_std
      value: -8.301679431909838
    - type: nauc_precision_at_1000_diff1
      value: -53.60779543720625
    - type: nauc_precision_at_1000_max
      value: 32.64507859011173
    - type: nauc_precision_at_1000_std
      value: 43.485127592553226
    - type: nauc_precision_at_100_diff1
      value: -15.707110225209581
    - type: nauc_precision_at_100_max
      value: 5.283790412535905
    - type: nauc_precision_at_100_std
      value: 65.56807579541196
    - type: nauc_precision_at_10_diff1
      value: 30.255881288208425
    - type: nauc_precision_at_10_max
      value: -3.151809768915205
    - type: nauc_precision_at_10_std
      value: 1.0480908087687508
    - type: nauc_precision_at_1_diff1
      value: 33.14382151181415
    - type: nauc_precision_at_1_max
      value: -25.77860726473143
    - type: nauc_precision_at_1_std
      value: -24.934811974694586
    - type: nauc_precision_at_20_diff1
      value: 15.164332420546184
    - type: nauc_precision_at_20_max
      value: -3.3790673721329565
    - type: nauc_precision_at_20_std
      value: 6.504278963012011
    - type: nauc_precision_at_3_diff1
      value: 34.47250505052381
    - type: nauc_precision_at_3_max
      value: -4.7389843027735665
    - type: nauc_precision_at_3_std
      value: -14.939623890867102
    - type: nauc_precision_at_5_diff1
      value: 30.347847737292888
    - type: nauc_precision_at_5_max
      value: -1.4447278947127309
    - type: nauc_precision_at_5_std
      value: -0.9944682720981688
    - type: nauc_recall_at_1000_diff1
      value: -7.7844569453114145
    - type: nauc_recall_at_1000_max
      value: 21.38875173601636
    - type: nauc_recall_at_1000_std
      value: 72.93348812363625
    - type: nauc_recall_at_100_diff1
      value: 25.734255573237263
    - type: nauc_recall_at_100_max
      value: -8.438874909217208
    - type: nauc_recall_at_100_std
      value: 31.718718288079117
    - type: nauc_recall_at_10_diff1
      value: 50.07254010357042
    - type: nauc_recall_at_10_max
      value: -4.334311945926507
    - type: nauc_recall_at_10_std
      value: -13.974286279580125
    - type: nauc_recall_at_1_diff1
      value: 53.051363797042306
    - type: nauc_recall_at_1_max
      value: -17.088092365200094
    - type: nauc_recall_at_1_std
      value: -28.724504958117336
    - type: nauc_recall_at_20_diff1
      value: 39.59317819427355
    - type: nauc_recall_at_20_max
      value: -8.715277246047938
    - type: nauc_recall_at_20_std
      value: -10.961701613637638
    - type: nauc_recall_at_3_diff1
      value: 56.972561065669844
    - type: nauc_recall_at_3_max
      value: -6.947793667419637
    - type: nauc_recall_at_3_std
      value: -21.437512253387673
    - type: nauc_recall_at_5_diff1
      value: 49.81190215009194
    - type: nauc_recall_at_5_max
      value: -8.065101854659815
    - type: nauc_recall_at_5_std
      value: -14.724500642391844
    - type: ndcg_at_1
      value: 42.857
    - type: ndcg_at_10
      value: 27.724
    - type: ndcg_at_100
      value: 38.634
    - type: ndcg_at_1000
      value: 49.138
    - type: ndcg_at_20
      value: 28.914
    - type: ndcg_at_3
      value: 33.495999999999995
    - type: ndcg_at_5
      value: 30.605
    - type: precision_at_1
      value: 44.897999999999996
    - type: precision_at_10
      value: 23.061
    - type: precision_at_100
      value: 7.611999999999999
    - type: precision_at_1000
      value: 1.471
    - type: precision_at_20
      value: 18.468999999999998
    - type: precision_at_3
      value: 33.333
    - type: precision_at_5
      value: 28.98
    - type: recall_at_1
      value: 3.4000000000000004
    - type: recall_at_10
      value: 16.8
    - type: recall_at_100
      value: 47.126000000000005
    - type: recall_at_1000
      value: 78.848
    - type: recall_at_20
      value: 25.641000000000002
    - type: recall_at_3
      value: 7.282
    - type: recall_at_5
      value: 10.495000000000001
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ToxicConversationsClassification (default)
      revision: edfaf9da55d3dd50d43143d90c1ac476895ae6de
      split: test
      type: mteb/toxic_conversations_50k
    metrics:
    - type: accuracy
      value: 84.19921875
    - type: ap
      value: 24.59609212285811
    - type: ap_weighted
      value: 24.59609212285811
    - type: f1
      value: 66.93277579966312
    - type: f1_weighted
      value: 86.98266846194493
    - type: main_score
      value: 84.19921875
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB TweetSentimentExtractionClassification (default)
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
      split: test
      type: mteb/tweet_sentiment_extraction
    metrics:
    - type: accuracy
      value: 76.25353706847766
    - type: f1
      value: 76.62648601703297
    - type: f1_weighted
      value: 76.09163868635312
    - type: main_score
      value: 76.25353706847766
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB TwentyNewsgroupsClustering (default)
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
      split: test
      type: mteb/twentynewsgroups-clustering
    metrics:
    - type: main_score
      value: 62.28448114372374
    - type: v_measure
      value: 62.28448114372374
    - type: v_measure_std
      value: 1.0068102803847288
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB TwitterSemEval2015 (default)
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
      split: test
      type: mteb/twittersemeval2015-pairclassification
    metrics:
    - type: cosine_accuracy
      value: 86.32651844787506
    - type: cosine_accuracy_threshold
      value: 75.02336502075195
    - type: cosine_ap
      value: 75.05733548730825
    - type: cosine_f1
      value: 68.78661087866108
    - type: cosine_f1_threshold
      value: 70.6207811832428
    - type: cosine_precision
      value: 62.885245901639344
    - type: cosine_recall
      value: 75.91029023746701
    - type: dot_accuracy
      value: 77.95195803778982
    - type: dot_accuracy_threshold
      value: 39931.048583984375
    - type: dot_ap
      value: 38.735233350262625
    - type: dot_f1
      value: 43.445570447858785
    - type: dot_f1_threshold
      value: 19839.00909423828
    - type: dot_precision
      value: 31.470518588681035
    - type: dot_recall
      value: 70.13192612137203
    - type: euclidean_accuracy
      value: 82.91708887166955
    - type: euclidean_accuracy_threshold
      value: 1185.8558654785156
    - type: euclidean_ap
      value: 63.499860584129465
    - type: euclidean_f1
      value: 60.93499444924139
    - type: euclidean_f1_threshold
      value: 1352.9718399047852
    - type: euclidean_precision
      value: 57.21565902246931
    - type: euclidean_recall
      value: 65.17150395778364
    - type: main_score
      value: 75.05733548730825
    - type: manhattan_accuracy
      value: 82.89324670680098
    - type: manhattan_accuracy_threshold
      value: 42541.79992675781
    - type: manhattan_ap
      value: 63.48742356371714
    - type: manhattan_f1
      value: 60.999877164967444
    - type: manhattan_f1_threshold
      value: 48992.218017578125
    - type: manhattan_precision
      value: 57.06734084118593
    - type: manhattan_recall
      value: 65.51451187335093
    - type: max_accuracy
      value: 86.32651844787506
    - type: max_ap
      value: 75.05733548730825
    - type: max_f1
      value: 68.78661087866108
    - type: max_precision
      value: 62.885245901639344
    - type: max_recall
      value: 75.91029023746701
    - type: similarity_accuracy
      value: 86.32651844787506
    - type: similarity_accuracy_threshold
      value: 75.02336502075195
    - type: similarity_ap
      value: 75.05733548730825
    - type: similarity_f1
      value: 68.78661087866108
    - type: similarity_f1_threshold
      value: 70.6207811832428
    - type: similarity_precision
      value: 62.885245901639344
    - type: similarity_recall
      value: 75.91029023746701
    task:
      type: PairClassification
  - dataset:
      config: default
      name: MTEB TwitterURLCorpus (default)
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
      split: test
      type: mteb/twitterurlcorpus-pairclassification
    metrics:
    - type: cosine_accuracy
      value: 89.12368533395428
    - type: cosine_accuracy_threshold
      value: 70.69376707077026
    - type: cosine_ap
      value: 86.17860530179777
    - type: cosine_f1
      value: 78.50265789375861
    - type: cosine_f1_threshold
      value: 67.95892119407654
    - type: cosine_precision
      value: 75.89304966578021
    - type: cosine_recall
      value: 81.29812134277795
    - type: dot_accuracy
      value: 82.56684907051655
    - type: dot_accuracy_threshold
      value: 18519.320678710938
    - type: dot_ap
      value: 69.25716425548741
    - type: dot_f1
      value: 65.25211505922167
    - type: dot_f1_threshold
      value: 15583.798217773438
    - type: dot_precision
      value: 58.211568651129085
    - type: dot_recall
      value: 74.23005851555283
    - type: euclidean_accuracy
      value: 85.91221329607637
    - type: euclidean_accuracy_threshold
      value: 1134.421157836914
    - type: euclidean_ap
      value: 78.89589179825943
    - type: euclidean_f1
      value: 70.8362823617061
    - type: euclidean_f1_threshold
      value: 1236.720848083496
    - type: euclidean_precision
      value: 68.61514036227177
    - type: euclidean_recall
      value: 73.20603634123806
    - type: main_score
      value: 86.17860530179777
    - type: manhattan_accuracy
      value: 85.91997516202895
    - type: manhattan_accuracy_threshold
      value: 40838.018798828125
    - type: manhattan_ap
      value: 78.87897330493252
    - type: manhattan_f1
      value: 70.81628097328672
    - type: manhattan_f1_threshold
      value: 44670.92590332031
    - type: manhattan_precision
      value: 68.51691864650829
    - type: manhattan_recall
      value: 73.2753310748383
    - type: max_accuracy
      value: 89.12368533395428
    - type: max_ap
      value: 86.17860530179777
    - type: max_f1
      value: 78.50265789375861
    - type: max_precision
      value: 75.89304966578021
    - type: max_recall
      value: 81.29812134277795
    - type: similarity_accuracy
      value: 89.12368533395428
    - type: similarity_accuracy_threshold
      value: 70.69376707077026
    - type: similarity_ap
      value: 86.17860530179777
    - type: similarity_f1
      value: 78.50265789375861
    - type: similarity_f1_threshold
      value: 67.95892119407654
    - type: similarity_precision
      value: 75.89304966578021
    - type: similarity_recall
      value: 81.29812134277795
    task:
      type: PairClassification
tags:
- mteb
---
