frame_question = """
Economic: costs, benefits, or other financial implications.
Capacity and resources: availability of physical, human or financial resources, and capacity of current systems.
Morality: religious or ethical implications.
Fairness and equality: balance or distribution of rights, responsibilities, and resources.
Legality, constitutionality and jurisprudence: rights, freedoms, and authority of individuals, corporations, and government.
Policy prescription and evaluation: discussion of specific policies aimed at addressing problems.
Crime and punishment: effectiveness and implications of laws and their enforcement.
Security and defense: threats to welfare of the individual, community, or nation.
Health and safety: health care, sanitation, public safety
Quality of life: threats and opportunities for the individual's wealth, happiness, and well-being.
Cultural identity: traditions, customs, or values of a social group in relation to a policy issue.
Public opinion: attitudes and opinions of the general public, including polling and demographics.
Political: considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters.
External regulation and reputation: international reputation or foreign policy of a country or
Other: any coherent group of frames not covered by the above categories.
"""

suffix_withimage = """
Given these definitions, output your response in the format below with exactly 3 fields:

'{"imagetextunderstanding": "The image shows <what it shows>. Text mentions <what it mentions in detail>. Text talks about <what it is about>. Hence, from the given frames, the frame is only <frame>.",
"imagetextframe": "<frame>",
"imagetextframereasoning": "imagetextframe is <frame> because <reasoning>"}'

<frame> should be one of: Economic, Capacity and resources, Morality, Fairness and equality, Legality, constitutionality and jurisprudence, Policy prescription and evaluation, Crime and punishment, Security and defense, Health and safety, Quality of life, Cultural identity, Public opinion, Political, External regulation and reputation, Other.

Ensure you have 3 fields filled in the JSON output:
"imagetextunderstanding", "imagetextframe", "imagetextframereasoning".
If you have less than 3 fields, you may have missed a field or reasoning.


"""

# Example:


suffix_noimage = """

Given these definitions, output your response in the format below with exactly 3 fields:

'{"textunderstanding": "Text mentions <what it mentions>. Text talks about <what it is about>. Hence, it seems to talk about <what it is about>. Hence, from the given frames, the frame is only <frame>.",
"textframe": "<frame>",
"textframereasoning": "textframe is <frame> because <reasoning>"}'

<frame> should be one of: Economic, Capacity and resources, Morality, Fairness and equality, Legality, constitutionality and jurisprudence, Policy prescription and evaluation, Crime and punishment, Security and defense, Health and safety, Quality of life, Cultural identity, Public opinion, Political, External regulation and reputation, Other.

Ensure you have 3 fields filled in the JSON output:
"textunderstanding", "textframe", "textframereasoning".
If you have less than 3 fields, you may have missed a field or reasoning.


"""
# Example:
# '{"final_frame": "Political", "final_frame_reasoning": "The text discusses the policies based on the people."}'

# suffix_withimage = """
# Given these definitions, output your response in the format below with exactly 7 fields:

# '{"image_understanding": "The image shows <what it shows>. When seen with text, the image seems to be about <what it is about>. Hence, the frame is <frame>.",
# "image_frame": "<frame>", 
# "image_reasoning": <reasoning of the frame based on the image in the context of the text>, 
# "text_frame": "<frame>", 
# "text_frame_reasoning": "The text discusses..<what it discusses>. Hence, the frame is <frame>.",
# "final_frame": "<frame>", 
# "final_frame_reasoning": "The image shows.. <what it shows> and the text discusses <what it discusses>. Hence, the frame is <frame>."}'

# <frame> should be one of: Economic, Capacity and resources, Morality, Fairness and equality, Legality, constitutionality and jurisprudence, Policy prescription and evaluation, Crime and punishment, Security and defense, Health and safety, Quality of life, Cultural identity, Public opinion, Political, External regulation and reputation, Other.
# Note:
# image_understanding - Consider what we see in the image and use it to determine the frame.
# image_frame - Consider what we see in the image when seen with text.
# image_reasoning - Start with "When seen with text, the image seems to be about.." and explain why you chose the frame based on the image.
# text_frame - Consider the text only.
# text_frame_reasoning - Start with "The text discusses..." and explain why you chose the frame based on the text.
# final_frame - frame of the image and text when both are shown together. Consider how the image and text relate to each other.
# final_frame_reasoning - Explain why you chose the frame based on the image and text together.

# Ensure you have 6 fields filled in the JSON output:
# "image_frame", "image_reasoning", "text_frame", "text_frame_reasoning", "final_frame", "final_frame_reasoning".

# If you have less than 7 fields, you may have missed a field or reasoning.

# """



# suffix_withimage = """
# Given these definitions, output your response in the format below with exactly 8 fields:

# '{"imageunderstanding": "The image shows <what it shows>. When seen with text, the image seems to be about <what it is about>.",
# "imageframe": "<frame>", 
# "imageframereasoning": "imageframe is <frame> because <reasoning>.",
# "textframereasoning": "The text discusses about..<what it discusses>. Hence, the frame is only <frame>.",
# "textframe": "<frame>", 
# "combinedframereasoning": "Since the image shows.. <what it shows> and the text discusses <what it discusses>. Hence, it seems to talk about <what it could be together>. Hence, the frame is only <frame>.",
# "combinedframe": "<frame>",
# "imagetextframereasoning": "combinedframereasoning is <frame> because <reasoning>"}'

# <frame> should be one of: Economic, Capacity and resources, Morality, Fairness and equality, Legality, constitutionality and jurisprudence, Policy prescription and evaluation, Crime and punishment, Security and defense, Health and safety, Quality of life, Cultural identity, Public opinion, Political, External regulation and reputation, Other.

# Ensure you have 8 fields filled in the JSON output:
# "imageunderstanding", "imageframe", "imageframereasoning", "textframereasoning", "textframe", "combinedframereasoning", "imagetextframe", "imagetextframereasoning".
# If you have less than 8 fields, you may have missed a field or reasoning.


