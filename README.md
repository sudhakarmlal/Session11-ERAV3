# Odia BPE  Tokenizer

Here we are going to build a Odia Tokennizer for inputing text into LLMs.

# Trainin Logs:
num_merges :  4908
89
90
91
92
93
..
...
...
...
4012

4013

4014

4015

4016

4017

4018

4019

4020

4021

4022

4023

4024

4025

4026

4027

4028

4029

4030

4031

4032

4033

4034

4035

4036

4037

4038

4039

4040

4041

4042

4043

4044

4045

4046

4047

4048

4049

4050

4051

4052

4053

4054

4055

4056

4057

4058

4059

4060

4061

4062

4063

4064

4065

4066

4067

4068

4069

4070

4071

4072

4073

4074

4075

4076

4077

4078

4079

4080

4081

4082

4083

4084

4085

4086

4087

4088

4089

4090

4091

4092

4093

4094

4095

4096

4097

4098

...
....
...

4993
4994
4995
4996
Decoded: ଏହା ବହୁତ ସାଧାରଣ ହୋଇଗଲାଣି ।
Compression ratio: 2.36

Vocabulary size: 5000
Number of merges: 4908

## Base Vocabulary
-  List all the unicode characters for Odia Language
-  Include all the vowels, consonants, signs, dependent vowels in the list
-  Make this whole list as base vocabulary

## Preprocess
- Collect the odia corpus data from huggingface or wikipedia
- I have taken dump from huggingface datasets for training. But during development I was using odia wikipedia data
- Make a regex pattern to separate odia words including spaces
    ```
    r""" ?[\u0B00-\u0B7F]+| ?[^\s]+|\s+(?!\S)|\s+"""
    ```
- Then input this word to train method using Byte-Pair-Encoding algorithm

## Train
- This method will take the separated words from preprocess step.
- It will pair the characters using BPE
- It will use the get_stats method to pair and find the maximum occuring pairs.
- After finding the most common pair, it will call the merge method to merge the pair with a new token.
- Finally vocabulary gets updated.

## Inference
- It has got encode method which can take the input text and provide the BPE codes
- It has also got decode method which will show the inputted text from BPE codes

## Results
- It has achieved compression ratio of more than 3.2 in many input text.
- It can show the same compression for new data as well.
- But It fails to achive this compression ratio in some cases as well.
- It has resullted 5000 as output tokens consisting the new vocabulary.

## Deployment
- This is deployed in HUggingface Spaces. The link is provided below.

    https://huggingface.co/spaces/satyanayak/odia_tokenizer_5k

- This can generate the BPE code as we type into input area dynamically
