# bank

원래는 python보다 R이 더 익숙하기 때문에 사용하는데 어려움이 있었음
결과적으로 R이 python보다 더 사용이 쉽고 빠르다는 것이 결론

데이터 전처리를 하는대 대부분의 시간을 사용함 
numpy와 sklearn에 전처리하는 모듈들도 있다는 것을 알게됨

categorical 데이터를 컬럼의 0,1의 값으로 변환하여 보다 알고리즘의 정확도를 높임

ex)[single] ->  married, divorced, single 
                    0        0        1

countineous 데이터는 min, max 기준으로 0~1사이 값으로 만듬

ex) (x - min)/(max - min)

hierarchy clustering을 통해서 대체적으로 몇개의 그룹으로 나뉘는지 보려했지만 메모리를 많이 써서 실패함
일단 kmean을 통해서 적당히 여러개의 k값을 정한후에 clustering을 시도해보고 적당한 것을 선정 

어느정도 well clust됬는지에 대한 결과를 알고 싶었지만 추후에 따로 cluster center와 데이터들 사이에 거리값의 평균이 최소가 되는 k를 선택하는것이 적달하다고 생각함 (여기까지는 구현하지 못함)

다시 나머지 변수들을 category와 countineous로 나누어 데이터를 전처리하고 모델을 만들어서 학습함 

사용한것은 decision tree, random forest, extra tree(정확한 알고리즘은 모르겠음)임

위 알고리즘을 선택한 이유는 decision tree류의 알고리즘이 category와 countineous로 복합된 데이터를 다루기에 가장 적합하기 때문

decision tree의 경우 0.65의 score가 나왔고 시험적으로 해보았음

앙상블기법을 사용한 다른 알고리즘이 더 잘나옴

random forest의 경우에는 모델의 수를 늘려서 vote하는 경우 10에서 50까지는 성능이 향상되나 그이상에서는 많이 향상되지 않음 100으로 하였을경우 0.83을 기록함 

extra tree의 경우에는 50과 100에서 큰차이 없이 대략 0.8이 나옴 

cross_val_score란 함수를 의심 없이 사용하였지만, train 데이터와 test데이터를 나눠서 모델의 성능을 측정해주는 함수

시간이 더 된다면 Gradient Boosting이나 Ada boost도 자세히 해보고 싶은 아쉬움이 남음 

