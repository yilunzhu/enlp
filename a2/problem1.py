import codecs

file = codecs.open("brown_vocab_100.txt", "r", "utf-16")
word_index_dict = {}

for i, line in enumerate(file):
    word_index_dict[line.strip()] = i

fw = codecs.open("word_to_index_100.txt", "w", "utf-8")
fw.write(str(word_index_dict))
file.close()
fw.close()

print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
