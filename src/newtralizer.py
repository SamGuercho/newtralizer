from typing import Dict
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from prompts.prompt_builder import NewtralizePromptBuilder
from prompts.prompts_enum import Prompts


class Newtralizer:
    def __init__(self, model_name='facebook/bart-base', device='cpu'):
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.prompt_builder = NewtralizePromptBuilder()

    def preprocess(self, combined_text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess the combined texts.
        :param combined_text: A single string combining all documents.
        :return: Tokenized texts.
        """
        tokenized_text = self.tokenizer(combined_text, return_tensors='pt', truncation=True, padding='max_length',
                                        max_length=1024)
        return tokenized_text

    def summarize(self, documents, max_length=150, min_length=40) -> str:
        """
        Generate a single neutral summary for the input documents.
        :param documents: List of strings, where each string is a document.
        :param max_length: Maximum length of the summary.
        :param min_length: Minimum length of the summary.
        :return: Single combined summary.
        """
        combined_text = self.prompt_builder.build_prompt(documents)
        tokenized_text = self.preprocess(combined_text)

        summary_ids = self.model.generate(
            tokenized_text['input_ids'].to(self.device),
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def evaluate(self, summary):
        """
        (Optional) Evaluate the neutrality of the summary.
        :param summary: The summary to evaluate.
        :return: Evaluation metrics (e.g., sentiment scores).
        """
        # Placeholder for evaluation logic, could integrate sentiment analysis tools
        pass


# Example usage
if __name__ == "__main__":
    documents = [
        """Title: Electric Scooters: A Green Solution for Urban Mobility

Byline: Jane Doe, Urban Technology Correspondent

Electric scooters have swiftly become a popular mode of transportation in cities worldwide. Advocates highlight the numerous benefits of these eco-friendly devices, which offer a practical alternative to cars and public transport.

One of the most significant advantages of electric scooters is their positive impact on the environment. Unlike gas-guzzling cars, scooters produce zero emissions, contributing to cleaner air and a reduction in urban pollution. This aligns with the growing global emphasis on sustainability and green technology.

Moreover, electric scooters provide a convenient solution for the "last mile" problem in urban commuting. They allow commuters to easily travel short distances from public transport hubs to their final destinations, reducing the reliance on cars and thereby easing traffic congestion.

The ease of use and affordability of electric scooters make them accessible to a broad range of people. Companies like Lime and Bird have made renting scooters as simple as using a smartphone app, and the cost is often lower than that of traditional taxis or ride-sharing services.

As cities continue to grapple with the challenges of urban mobility, electric scooters offer a promising and sustainable solution that benefits both individuals and the community at large.""",
        """Title: The Dark Side of Electric Scooters: A Menace to Urban Safety

Byline: John Smith, Public Safety Reporter

While electric scooters are often touted as a modern solution to urban transportation woes, their rapid proliferation has brought significant safety and public space concerns.

One of the primary issues with electric scooters is the safety risk they pose to both riders and pedestrians. Hospitals have reported a surge in injuries related to scooter accidents, many involving head trauma and fractures. Unlike bicycles, scooters often lack robust safety features such as lights and bells, making them hazardous, especially in low-light conditions.

Furthermore, the way scooters are used and parked can create chaos in public spaces. Scooters are frequently abandoned on sidewalks, obstructing pathways and posing a tripping hazard. This has sparked frustration among pedestrians, particularly those with disabilities who find their mobility further restricted.

Cities are also struggling to regulate scooter usage effectively. Many users disregard traffic laws, riding on sidewalks, in pedestrian zones, and against the flow of traffic. This anarchy contributes to an environment where accidents are more likely to occur, endangering both riders and non-riders alike.

While the convenience of electric scooters cannot be denied, the associated safety and public space issues raise serious questions about their place in urban transportation networks. Stricter regulations and better infrastructure are needed to mitigate these problems and ensure the safety of all city dwellers."""
    ]

    newtralizer = Newtralizer(model_name='facebook/bart-large-cnn',
                              device='cuda' if torch.cuda.is_available() else 'cpu')
    summary = newtralizer.summarize(documents)

    for i, summary in enumerate(summary):
        print(f"Summary {i + 1}: {summary}")