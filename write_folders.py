from shutil import copyfile

# Set or words to train on
WORDS = set([',','the','of','to'])
# Or, set FULL_SET to True to use the full set of data
FULL_SET = False


def copy_words():
     with open("words.txt") as fp:
      for line in fp:
           word = line.split(' ')[8][:-1] # chop of the last char, which is always newline
           if FULL_SET or word in WORDS:
              # add actual word to labels
              fn = line.split(' ')[0]
              copyfile("words/" + fn +'.png', word+"/" + fn +'.png') 
def count_words():
     count = 0
     split = 0.1
     with open("words.txt") as fp:
      for line in fp:
           word = line.split(' ')[8][:-1] # chop of the last char, which is always newline
           if FULL_SET or word in WORDS: 
              count += 1
     print "Total count is", count
     print "For split of", split, "use", count * (1-split), "for training and", count*split, "for validation"

if __name__ == '__main__':
  #copy_words()
  count_words()