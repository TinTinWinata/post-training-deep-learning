Runtime pada google collab dapat diganti ke local -> Runtime -> Change Type

Library yang akan dipakai :

- Tensorflow
- Keras

SLP dan MLP Model

SLP (Single Layer Perception) adalah model paling sederhana dimana hanya mampu mempelajari separable patterns secara linear.[4] SLP cukup bermanfaat untuk membedakan input menjadi 2 class. Hampir mirip dengan K-Means namun jauh lebih sederhana karena hanya melibatkan dua buah class saja. Dapat dikatakan untuk sebuah metode yang dapat digolongkan sebagai AI, memang SLP terkesan kurang cerdas.

MLP (Multi Layer Perception) merupakan pengembangan dari SLP dimana numlah neuron yang semakin banyak membuat banyak perhitungan yang harus dikerjakan pada setiap layer. Akibatnya weighted sum dan function activation pun akan semakin kompleks. Penambahan layer ini terjadi pada hidden layer. Hidden layer pada MLP dapat mengandung beberapa hidden layer lainnya. MLP ini juga yang menjadi cikal bakal metode deep learning. Sebagai ilustrasi, perhatikan diagram MLP dibawah.

refrensi : Link (https://sistem-komputer-s1.stekom.ac.id/informasi/baca/Mengenal-Perbedaan-Artificial-Intelligence-Machine-Learning-Neural-Network-Deep-Learning-Seri-2/422237776eebac1e9ce55eb11b9635704dfe1507)

Perbedaan SLP & MLP

- SLP (Single Layer Perception)
  -> Input langsung ke output
  -> Dapat lienar separable

- MLP (Multi Layer Perception)
  -> Terdapat hidden layer pada input baru bisa ke output.
  -> Dapat non-linear

Tips :

- Train Data wajib di Shuffle (Datanya akan terlihat acak)

Terdapat 3 hal yang dilakukan untuk memanipulasi data :

- Training
- Data Validation
  -> Dilakukan saat training (melakukan check pada training)
  -> Yang kita lihat : Score utk Training, Score utk Validation
- Test Validation
  -> Predict untuk melihat dengan data yang asil nya

Di keras terdapat 2 cara membuat model :

- Sequential
  -> Input nya terurut cth. (Input -> Hidden 1 -> Hidden 2 -> Output)
  -> Bisa di convert ke functional

- Functional
  -> Kita dapat menentukan urutannya
  -> Tidak dapat di convert ke sequential

Cara mereka dibuat sama saja, yang perbedaan cuma pemanggilan hubungan layer nya yang mana.

Dense = Fully connection layer
Loss
-> BCE Binary cross entropy
-> CCE Categorical cross entropy

Referensi : https://shiva-verma.medium.com/understanding-different-loss-functions-for-neural-networks-dd1ed0274718


Referensi Dense Keras : 
https://keras.io/api/layers/core_layers/dense/

Activation dalam hidden layer biasa menggunakan 'relu' sedangkan output 'sigmoid' 

Overfitting is an undesirable machine learning behavior that occurs when the machine learning model gives accurate predictions for training data but not for new data


Flattening is used to convert all the resultant 2-Dimensional arrays from pooled feature maps into a single long continuous linear vector

Dataset gambar
- Cifar10
- Cifar100

https://paperswithcode.com/dataset/cifar-100
https://paperswithcode.com/dataset/mnist
https://paperswithcode.com/dataset/cifar-10


NLP - Berd

Menggunakan Embedding
Embedding -> Mengambil dari kalimat kalimat dan menaruh di vektor yang berdekatan

Harus di tokenize dulu 
Dari kalimat > Kata Kata

method TextVectorization berati langsung tokenize + embedding
stopword and lemmatization sudah tidak dipakai, karena penting untuk attention

Link Google Collab : 
- Perceptron s/d CNN
https://colab.research.google.com/drive/1pPMB0hgIZFaj41y38fECQuH_se9Yyagl?usp=sharing
- Embedding
https://colab.research.google.com/drive/1TwHwhaCII9Ra-FUXhvo8lDdylgt3AqW5?usp=sharing 
