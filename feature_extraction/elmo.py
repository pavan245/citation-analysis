from allennlp.modules.elmo import Elmo, batch_to_ids

weights_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'

elmo = Elmo(options_file, weights_file, 1, dropout=0)
text = ['Backgammon', 'is', 'one', 'of', 'the', 'oldest', 'known', 'board', 'games']

batch = batch_to_ids(text)
print(batch)

dict = elmo.forward(batch)

print(dict['elmo_representations'])