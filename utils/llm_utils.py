import json
import os
from typing import Dict

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

import utils.journal_query as jq
import utils.prompt_templates as pt
from utils.parse_sahha_score import main
from utils.prompt_templates import *

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_db_context(user_chat):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )  # this should be abstracted later
    db = jq.get_db(embeddings)
    
    ## Commenting out for now, just going to do a search with the user's chat input
    prompt = pt.get_topics_from_user_chat()
    model = ChatGoogleGenerativeAI(model='gemini-pro')
    chain = LLMChain(llm=model, prompt=prompt)
    topics = chain.run({"user_chat": user_chat})

    docs, sims = jq.get_docs_with_query(db, topics, num_of_docs=4, score_threshold=0)
    db_context_string = jq.format_docs(docs, sims)
    return db_context_string


def get_llm_instance():
    model = genai.GenerativeModel("gemini-1.5-flash")
    # model = genai.GenerativeModel("gemini-1.0-pro")
    return model


def get_completion(model, prompt):
    response = model.generate_content(prompt)
    return response.text


async def async_get_completion(model, prompt):
    response = await model.generate_content_async(prompt)
    return response.text


def get_sahha_insights(date_time, user_id):
    """
    Integration of Sahha API is not possible now so as aligned, we have retrieved a static json file
    for a given day from the Sahha team. We assume user_id to be 4 and the output is from a given day.
    """
    user_id = 4  # hard-coded
    random_user_json, dic = main()

    with open("data/sahha_metadata_flatten.json") as f:
        md = json.load(f)

    lst1 = dic["activity_scores"]
    lst2 = dic["sleep_scores"]
    factor_lst = lst1 + lst2

    factors_df = pd.DataFrame(factor_lst)  # sophie_dic['factors'])

    lacking_df = factors_df[factors_df["state"] == "low"]
    lacking_df["metadata"] = lacking_df["name"].map(md)

    lacking_df["prompt"] = lacking_df.apply(
        lambda x: f"For {x['name']}, {x['metadata']}, you're at {x['value']} {x['unit']}, "
        f"which is {x['state']}, compared to the average of {x['score']} {x['unit']}.",
        axis=1,
    )

    sahha_prompt = " ".join(lacking_df["prompt"].tolist())

    well_being_score = dic["user"]["score"]

    return sahha_prompt, well_being_score
