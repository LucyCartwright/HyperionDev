import spacy

nlp = spacy.load('en_core_web_md')

# Compare 'cat', 'monkey', 'banana'
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print("\nComparing 'cat', 'monkey', 'banana':\n")
print("'cat' similarity to 'monkey':", word1.similarity(word2))
print("'banana' similarity to 'monkey':", word3.similarity(word2))
print("'banana' similarity to 'cat':", word3.similarity(word1))


# Compare 'cat', 'apple', 'monkey', 'banana'
print("\nComparing 'cat', 'apple, 'monkey', 'banana':\n")
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# Short sentences comparison
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
print("\nComparing short sentences to 'Why is my cat on the car':\n")
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# spaCy computes the similarity between 'cat' and 'monkey' to be 0.59 (rounded to 2dp)
# These two are much more similar than 'cat' and 'banana' (0.22, 2dp). 'Cat' and 'monkey'
# are both mammals, whereas 'banana' is a fruit. spaCy finds a higher degree of similarity
# between 'monkey' and 'banana' (0.40, 2dp) than 'cat' and 'banana', I imagine because of
# the common understanding that monkeys eat bananas.

# Testing a similar triad: 'owl', 'happy', 'wise'
word1 = nlp("owl")
word2 = nlp("happy")
word3 = nlp("wise")
print("\nComparing 'owl', 'happy', 'wise':\n")
print("'owl' similarity to 'happy':", word1.similarity(word2))
print("'owl' similarity to 'wise':", word1.similarity(word3))
print("'wise' similarity to 'happy':", word3.similarity(word2))

# Both 'happy' and 'wise' are adjectives used to describe humans and there is a modest degree
# of similarity between them (0.36, 2dp). There is minimal similarity between 'owl' (a bird)
# and 'happy' (0.03, 2dp), but there is a higher degree of similarity between 'owl' and 'wise'
# (0.07, 2dp). I imagine that this is because of the common perception that owls are wise
# birds, e.g. the association of an owl with Athena, the Greek goddess of wisdom



# =====================================================

# Loading simpler language model 'en_core_web_sm' to compare with output from 'example.py' file
nlp = spacy.load('en_core_web_sm') 

# List of six complaints
complaints = [ 'We bought a house in  CA. Our mortgage was handled by a company called ki. Soon after the mortgage was sold to ABC. Shortly after that XYZ took over the mortgage. The other day we got a notice not to send our payment to them but to loi instead. This is all so frustrating and wreaks of the  mortgage nightmare.',
'I got approved for a loan to buy a house I have submitted everything I need to for them I paid for the inspection and paid good faith check after all of that they said I did not get approved for the loan to cancel my contract because they do not want to wait for the down payments assistant said that the Sellers do not want to wait that long I feel like they are getting over on me I feel that they should have told me that I did not get approved before I spent my money and picked out a house Carrington mortgage in Ohio ',
'As per the correspondence, I received from : The University  This is to inform you that I have recently pulled my credit report and noticed that there is a collection listing from The University  on my credit report. I WAS never notified of this collection action or that I owed the debt. This letter is to inform you that I would like a verification of the debt and juilo ability to collect this money from me.',
'I am writing to dispute the follow information in my file.ON BOTH TransUnion & . for {$15000.00}. I have contacted this agency to advise to STOP CALLING ME this case was dismissed in court  2014. Please see the attached document from  County State Court. Thanking you in advanced regarding this matter.',
'I have not had a XXXX phone since early 2007. I have tried to resolve my bill in the past but it keeps reposting an old bill. I have no way to provide financial info from 8 years ago and they know that so they want me to prove it to them but I have no way to do that. Is there anyway to get  to find out how old it is.',
'I posted dated a check and mailed it for 2015 for my mortgage payment as my mortgage company will only take online payments if all the late charges are paid at once ( also illegal ), and the check was cashed on 2015 which cost me over {$70.00} in over draft fees with my bank.'
]

# Compare the similarity of the complaints
print("\n-------------Complaints similarity---------------")
for token in complaints:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))

# List of six recipe instructions
recipes= [ 'Bake in the preheated oven, stirring every 20 minutes, until sugar mixture has baked and caramelized onto popcorn and cashews, about 1 hour. Spread cashew caramel corn onto a parchment paper-lined baking sheet to cool. If desired, form into balls while still warm.',
'Combine brown sugar, corn syrup, butter, salt, and cream of tartar in a large saucepan. Bring to a boil, stirring constantly, until a candy thermometer inserted into the middle of the syrup, not touching the bottom, reads 260 degrees F (127 degrees C), 6 to 8 minutes.',
'Lift marshmallow fudge out of the pan by the edges of the foil and place on a large cutting board. Dip a large knife in the remaining confectioners\' sugar and slice fudge into 1 1/2-inch squares, continually dipping knife in the sugar after each slice.',
'Melt butter in a medium saucepan over medium heat; stir in condensed milk. Pour in chocolate chips; cook and stir until melted, 5 to 10 minutes.',
'Lightly grease a cookie sheet. Deflate the dough and turn it out onto a lightly floured surface. Roll the marzipan into a rope and place it in the center of the dough. Fold the dough over to cover it; pinch the seams together to seal. Place the loaf, seam side down, on the prepared baking sheet. Cover with a damp cloth and let rise until doubled in volume, about 40 minutes. Meanwhile, preheat oven to 350 degrees F (175 degrees C)',
'In a large bowl, cream together the butter, brown sugar, and white sugar. Beat in the instant pudding mix until blended. Stir in the eggs and vanilla. Blend in the flour mixture. Finally, stir in the chocolate chips and nuts. Drop cookies by rounded spoonfuls onto ungreased cookie sheets.'
]

# Compare the similarity of the recipes
print("\n-------------Recipes similarity---------------")
for token in recipes:
    token = nlp(token)
    for token_ in recipes:
        token_ = nlp(token_)
        print(token.similarity(token_))


# Loop through every recipe instruction and compare it with a complaint.
print("\n-------------Recipes vs complaints similarity---------------")

for token in recipes:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))


# Running this code yields the following warning message:

# "UserWarning: [W007] The model you're using has no word vectors loaded, so
# the result of the Doc.similarity method will be based on the tagger,
# parser and NER, which may not give useful similarity judgements. This may happen
# if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship
# with word vectors and only use context-sensitive tensors. You can always add your
# own word vectors, or use one of the larger models instead if available"

# The degrees of similarity can be compared with the outputs of the same script
# run using the model 'en_core_web_md'. The calculated degrees of similarity using
# the simpler 'sm' model are lower than those calculated using the 'md' model.
# I've compared the two outputs in Excel and the'sm' degrees of similarity are
# on average 28% less than the degrees calculated using 'md'. This demonstrates
# the importance of using a model with word vectors, such as 'en_core_web_md'.