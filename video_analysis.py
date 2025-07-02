import numpy as np
import json
import os
import random
import time
import boto3
import requests
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions, CategoriesOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from decouple import config

# Accessing the environment variables stored in .env file
AWS_ACCESS_KEY_ID = config('aws_access_key_id')
AWS_SECRET_KEY = config('aws_secret_key')
MY_REGION = config('my_region')
BUCKET_NAME = config('bucket_name')
LANG_CODE = config('lang_code')
IBM_APIKEY = config('ibm_apikey')
IBM_URL = config('ibm_url')

# Authenticate Watson Tone Analyzer
apikey = "PYbJUuBdyWB_7mnXsNs2peBu0Gj6o9PnY80j0AFs2yDz"
url = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com"

authenticator = IAMAuthenticator(apikey)
nlu = NaturalLanguageUnderstandingV1(version='2022-08-10', authenticator=authenticator)
nlu.set_service_url(url)

try:
    response = nlu.analyze(
        text="Hello world, I love IBM Watson.",
        features=Features(emotion=EmotionOptions())
    ).get_result()
    print(response)
except Exception as e:
    print(f"Error: {e}")


# AWS S3 Resource Client
s3 = boto3.resource(service_name="s3", region_name=MY_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_KEY)

# Generate random job name for AWS Transcribe
def random_job_name():
    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    return ''.join(random.choices(chars, k=10))

# Transcribe audio using AWS Transcribe
def extract_text(file_name):
    try:
        s3.Bucket(BUCKET_NAME).upload_file(Filename=f"./static/{file_name}", Key=file_name)
    except Exception as e:
        print("Could not upload file to S3:", e)
        return "", {}

    transcribe = boto3.client("transcribe",
                              region_name=MY_REGION,
                              aws_access_key_id=AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_SECRET_KEY)

    job_name = file_name.split('.')[0] + "-" + random_job_name()
    job_uri = f"s3://{BUCKET_NAME}/{file_name}"

    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': job_uri},
            MediaFormat="webm",
            LanguageCode="en-IN"
        )
    except Exception as e:
        print("Error starting transcription job:", e)
        return "", {}

    # Wait for job completion
    while True:
        try:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            print("Waiting for transcription to complete...")
            time.sleep(10)
        except Exception as e:
            print("Error checking transcription status:", e)
            return "", {}

    if status['TranscriptionJob']['TranscriptionJobStatus'] == "FAILED":
        print("Transcription job failed:", status['TranscriptionJob']['FailureReason'])
        return "", {}

    try:
        transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        response = requests.get(transcript_url)
        if response.status_code != 200:
            print("Failed to download transcript JSON")
            return "", {}

        transcript_json = response.json()
    except Exception as e:
        print("Error retrieving transcript data:", e)
        return "", {}

    # Extract the actual text from the transcript
    try:
        transcripts = transcript_json.get("results", {}).get("transcripts", [])
        if transcripts:
            text = transcripts[0].get("transcript", "")
        else:
            text = ""
    except Exception as e:
        print("Error extracting transcript:", e)
        text = ""

    return text, transcript_json

# IBM Wats
def analyze_tone(text):
    try:
        response = nlu.analyze(
            text=text,
            features=Features(
                emotion=EmotionOptions(),
                sentiment=SentimentOptions()
            )
        ).get_result()

        # Extract only actual emotions from response
        emotions = response.get('emotion', {}).get('document', {}).get('emotion', {})

        # Convert values to percentages (rounded)
        return {emotion: round(score * 100) for emotion, score in emotions.items()}

    except Exception as e:
        print("Error analyzing tone with NLU:", e)
        return {}
