import pandas as pd
from bs4 import BeautifulSoup

def parse_sgm_to_dataframe(file_path: str) -> pd.DataFrame:
    # Open and read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        sgm_data = file.read()

    # Parse the SGML data
    soup = BeautifulSoup(sgm_data, 'html.parser')

    # List to hold parsed data
    data = []

    # Iterate over each Reuters tag in the SGML
    for reuters in soup.find_all('reuters'):
        # Extract the NEWID attribute to serve as an ID
        article_id = reuters.get('newid')

        # Extract the BODY content
        body = reuters.find('body')
        body_text = body.get_text().strip() if body else ''

        # Extract the TOPICS
        topics = reuters.find('topics')
        if topics:
            # Get all topics listed under <D> tags
            topics_list = [d.get_text().strip() for d in topics.find_all('d')]
            # If there are topics, add a row for each topic
            if topics_list:
                for topic in topics_list:
                    data.append({'ID': article_id, 'Topic': topic, 'Body': body_text})
            else:
                # If <topics> tag exists but is empty, add a row with empty string for Topic
                data.append({'ID': article_id, 'Topic': '', 'Body': body_text})
        else:
            # If there's no <topics> tag, add a row with None for Topic
            data.append({'ID': article_id, 'Topic': None, 'Body': body_text})

    # Create a DataFrame from the parsed data
    df = pd.DataFrame(data)
    return df