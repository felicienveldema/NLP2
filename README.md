# Projects for the Natural Language Processing 2 Course, University of Amsterdam
## Authors: Adriaan de Vries, Verna Dankers, Féliciën Veldema

### Project 2: Neural Machine Translation models for English-French sentences from Multi30k Task 1 dataset

Train a model using the following command:

```
python train.py --english_train training/train.en --french_train training/train.fr --english_valid val/val.en --french_valid val/val.fr --enc_type avg --dec_type gru --attention dot --lr 0.0005 --tf_ratio 0.75 --batch_size 32 --epochs 10 --dim 400 --num_symbols 10000 --min_count 1 --max_length 74 [--lower] [--enable_cuda]

```

We truecased the corpus by default. Set the ```lower``` flag to lowercase. If a GPU is available, set the ```enable_cuda``` flag.
For some arguments there are multiple options:
1. Encoder: avg | transformer | gru
2. Attention: dot | bilinear | multihead

Test a saved model using the following command:

```
python3 test.py --english test/test_2017_flickr.en --french test/test_2017_flickr.fr --encoder encoder_type=gru.pt --decoder decoder_type=gru.pt --corpus corpus.pickle --max_length 74 [--enable_cuda] [--transformer]
```

If testing with different decoders or encoders, replace the path names. If a GPU is available, set the ```enable_cuda``` flag. If testing a transformer conder, set the ```transformer``` flag, which is needed for visualization of the attention weights.

Examples for attention visualizations can be found in the Project2/visualization folder.
Our results on Multi30k testing data, En-Fr are the following:

<table class="tg">
  <tr>
    <th class="tg-e3zv">Encoder<br></th>
    <th class="tg-e3zv">Attention</th>
    <th class="tg-e3zv">BLEU</th>
    <th class="tg-e3zv">TER</th>
    <th class="tg-9hbo">METEOR</th>
  </tr>
  <tr>
    <td class="tg-031e">Averaging</td>
    <td class="tg-031e">Multihead</td>
    <td class="tg-031e">30.08</td>
    <td class="tg-031e">47.89</td>
    <td class="tg-yw4l">30.29</td>
  </tr>
  <tr>
    <td class="tg-031e">GRU</td>
    <td class="tg-031e">Multihead</td>
    <td class="tg-e3zv">33.81</td>
    <td class="tg-e3zv">44.60</td>
    <td class="tg-9hbo">32.10</td>
  </tr>
  <tr>
    <td class="tg-031e">Transformer</td>
    <td class="tg-031e">Multihead</td>
    <td class="tg-031e">31.21</td>
    <td class="tg-031e">51.55</td>
    <td class="tg-yw4l">31.22</td>
  </tr>
</table>

### Project 1: Evaluating IBM alignment models

For the first project we implemented IBM 1 and IBM 2, along with the Variational Bayes and Expectation Maximisation optimisation algorithms.
The models are trained using data from a parallel corpus. Code to read in the data can be found in the file ```data.py```.
Words occurring once are mapped to the token '-UNK-'.
Code to evaluate alignments found is given in the file ```aer.py```.
The models themselves are implemented in interactive jupyter notebooks:
1. ```IBM1-EM.ipynb```
2. ```IBM1-VB.ipynb```
3. ```IBM2-EM.ipynb```
4. ```IBM2-VB.ipynb```

Run the notebooks one by one to retrain the models. Performance on the testing data is presented in the NAACL format in the folder ```test_results```.

The following image presents an example alignment for IBM1 VB:
<img src="Project1/garbage_vb.png" />
The following image presents an example alignment for IBM1 EM:
<img src="Project1/garbage_em.png" />

The results on test data were the following:

<table class="tg">
  <tr>
    <th class="tg-us36">Model</th>
    <th class="tg-us36">Training</th>
    <th class="tg-us36">Selection</th>
    <th class="tg-us36">AER<br></th>
  </tr>
  <tr>
    <td class="tg-us36">IBM 1</td>
    <td class="tg-us36">MLE</td>
    <td class="tg-us36">AER</td>
    <td class="tg-us36">0.2852<br></td>
  </tr>
  <tr>
    <td class="tg-us36">IBM 1</td>
    <td class="tg-us36">VB</td>
    <td class="tg-us36">AER</td>
    <td class="tg-us36">0.2866</td>
  </tr>
  <tr>
    <td class="tg-us36">IBM 1</td>
    <td class="tg-us36">MLE</td>
    <td class="tg-us36">LL</td>
    <td class="tg-us36">0.2856</td>
  </tr>
  <tr>
    <td class="tg-us36">IBM 1</td>
    <td class="tg-us36">VB</td>
    <td class="tg-us36">ELBO</td>
    <td class="tg-us36">0.2863</td>
  </tr>
  <tr>
    <td class="tg-us36">IBM 2</td>
    <td class="tg-us36">MLE</td>
    <td class="tg-us36">AER</td>
    <td class="tg-us36">0.2068</td>
  </tr>
  <tr>
    <td class="tg-us36">IBM 2</td>
    <td class="tg-us36">VB</td>
    <td class="tg-us36">AER</td>
    <td class="tg-us36">0.2054</td>
  </tr>
  <tr>
    <td class="tg-us36">IBM 2</td>
    <td class="tg-us36">MLE</td>
    <td class="tg-us36">LL</td>
    <td class="tg-us36">0.2047</td>
  </tr>
  <tr>
    <td class="tg-us36">IBM 2</td>
    <td class="tg-us36">VB</td>
    <td class="tg-us36">ELBO</td>
    <td class="tg-us36">0.2036</td>
  </tr>
</table>
