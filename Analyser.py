import json

file = open("DataSets/trainingdata.txt", 'r')

dict = {}

common_words = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z", "between", "name",
                "making", "should", "there", "under", "because", "made", "than", "their", "take", "more", "show", "set",
                "our", "could", "will", "well", "two", "new", "with", "were", "did", "say", "out", "no", "are", "and",
                "or", "of", "to", "a",  "as", "also", "an", "at", "we", "u", "for", "from", "the", "if", "he",
                "\n", "then", "not", "from", "it", "cut", "about", "months", "other",
                "they", "any", "can", "its", "have", "had", "in", "on", "only", "but", "s", "up", "by", "now",
                "said", "be", "was", "been", "into", "all", "over", "who", "do",
                "what", "when", "which", "would", "his", "him", "has", "that", "while", "after", "today", "them"]

position = 0

for line in file:
    words = line.split(" ")
    classified = words[0]

    words = words[1:]

    for word in words:
        if word not in common_words:
            if word not in dict:
                dict[word] = {"position":position}
                position += 1
            if classified not in dict[word]:
                dict[word][classified] = 0
            dict[word][classified] += 1

new_dict = {}
i = 0
for key in dict:
    sum = 0
    for Key in dict[key]:
        if Key != "position":
            sum += dict[key][Key]
    if sum > 400:
        new_dict[key] = {"position": i}
        i += 1


outfile = open("DataSets/dataset50transpose.json", "w")
json.dump(new_dict, outfile, indent=4)
outfile.close()
