from openai import OpenAI
import time

from perspective import PerspectiveAPI
p = PerspectiveAPI("AIzaSyD3aJODD7HViVaAjt2vTrSvRzS3zOCsioE")
# SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, THREAT, SEXUALLY_EXPLICIT, and FLIRTATION
result = p.score("This is a comment", ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT", "SEXUALLY_EXPLICIT", "FLIRTATION"])
print("Toxicity score: ", result)

# OPENAI_API_KEY = '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'

# client = OpenAI(api_key=OPENAI_API_KEY)

# while True:
#     try:
#         response = client.moderations.create(input="Sample text goes here.")
#         break
#     except:
#         print('error; waiting ten seconds and resubmitting')
#         time.sleep(10)

# output = response.results[0]

# # get a score list
# score_list = [output.category_scores.harassment, output.category_scores.harassment_threatening, \
#               output.category_scores.hate, output.category_scores.hate_threatening, \
#               output.category_scores.self_harm, output.category_scores.self_harm_instructions, \
#               output.category_scores.self_harm_intent, output.category_scores.sexual, \
#               output.category_scores.sexual_minors, output.category_scores.violence, \
#               output.category_scores.violence_graphic]

# print(score_list)
