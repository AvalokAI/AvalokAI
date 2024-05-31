from tqdm import tqdm

from data.data import Data
from embed.chunk import Chunk
from embed.embed import Embed
from vectordb.vectordb import PineConeVectorDB


def main():

    chunker = Chunk()
    embedder = Embed()
    db = PineConeVectorDB(embedder.get_embedding_size())
    raw_data = Data("/home/sankalp/harvard_cold_cases")

    while True:
        query = input("Enter the query: ")
        query = query.strip()
        if query == "":
            break
        embedding = embedder.embed_single_text(query)
        matches = db.retrive_chunks(embedding, 10)
        # matches = [{"id": "2636981-1", "score": 123}]
        for i, match in enumerate(matches):
            print(f"Matching id {match['id']} score {match['score']}")
            opinion_id, chunk_id = tuple(map(int, match["id"].split("-")))
            doc = raw_data.get_document_from_id(opinion_id)
            splits = chunker.get_chunks([doc])
            split = splits[chunk_id]

            print(f"----------- Match {i} ----------")
            print(split)


if __name__ == "__main__":
    main()
