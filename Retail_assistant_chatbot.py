# RETAIL ASSISTANT CHATBOT
# Separate program linked to the Sales Regression model through the saved
# files created by the training script: sales_model.pkl and model_features.pkl
# Nothing in the training script or the prediction tool is changed.


import os
import difflib
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# 1. LOAD THE SAVED REGRESSION MODEL (same files used by the prediction tool)

folder = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(folder, "sales_model.pkl")
features_path = os.path.join(folder, "model_features.pkl")

if not os.path.exists(model_path):
    print("ERROR: sales_model.pkl was not found.")
    print("Run the training script first and keep the file in this folder.")
    input("Press Enter to close...")
    exit()

if not os.path.exists(features_path):
    print("ERROR: model_features.pkl was not found.")
    print("Run the training script first and keep the file in this folder.")
    input("Press Enter to close...")
    exit()

sales_model = joblib.load(model_path)
model_features = joblib.load(features_path)


# 2. TRAINING DATA FOR THE CHATBOT (intent: example questions)

training_data = {

    "increase_sales": [
        "how can i increase sales",
        "how do i improve revenue",
        "how can i boost sales",
        "how can i attract customers",
        "how do i grow my business",
        "how can i get more customers",
        "how can i sell more products",
        "how can i increase customer purchases",
        "how can i increase my revenue",
        "what can i do to increase sales"
    ],

    "low_sales": [
        "why are sales low",
        "what causes low sales",
        "why is revenue low",
        "sales are poor",
        "sales are decreasing",
        "why are my sales decreasing",
        "why am i getting fewer sales",
        "what can cause poor sales",
        "why is my business not making enough sales",
        "why have my sales dropped"
    ],

    "prediction": [
        "what is my prediction",
        "show my sales amount",
        "tell me the prediction",
        "what are projected sales",
        "what is the predicted sales amount",
        "show the predicted sales",
        "what sales amount is expected",
        "what are my expected sales",
        "can i see my sales prediction",
        # forecast and predict wording
        "make a sales forecast",
        "forecast my sales",
        "i want to forecast sales",
        "sales forecast",
        "sales forecasting",
        "predict my sales",
        "predict sales for a new transaction",
        "i want a sales prediction",
        "make a prediction",
        "forecast the sales amount",
        "can you forecast my sales",
        "can you predict my sales"
    ],

    "discount": [
        "tell me about discounts",
        "should i use discounts",
        "how do discounts affect sales",
        "are discounts good for sales",
        "should i offer discounts",
        "how can discounts increase sales",
        "can discounts reduce profit",
        "what is the effect of discounts",
        "how should i use discounts"
    ],

    "analysis": [
        "analyse sales",
        "sales analysis",
        "analyse my prediction",
        "analyze sales",
        "how can i analyse my sales",
        "how do i analyze sales",
        "help me analyze my sales",
        "what should i look at when analyzing sales",
        "how can i evaluate my sales"
    ],

    "profit": [
        "how can i improve profit",
        "increase profit",
        "improve profitability",
        "how can i make more profit",
        "how can i increase profitability",
        "how can i improve my business profit",
        "what can i do to increase profit",
        "how can i reduce unnecessary costs",
        "how can i make my business more profitable"
    ],

    "inventory": [
        "tell me about inventory",
        "inventory management",
        "how do i manage stock",
        "how should i manage inventory",
        "how can i manage my stock",
        "how do i avoid stock shortages",
        "how can i avoid overstocking",
        "how can i improve inventory management",
        "how do i monitor stock levels"
    ]
}

training_questions = []
training_labels = []

for intent_name, questions in training_data.items():
    for question in questions:
        training_questions.append(question)
        training_labels.append(intent_name)


# 3. TRAIN THE CHATBOT (bag-of-words + Multinomial Naive Bayes)

vectorizer = CountVectorizer()
X_chat = vectorizer.fit_transform(training_questions)

chat_model = MultinomialNB()
chat_model.fit(X_chat, training_labels)

# Words the chatbot knows (used for spelling correction)
vocabulary = vectorizer.get_feature_names_out().tolist()


# 4. PRE-WRITTEN RESPONSES

responses = {

    "increase_sales": (
        "SALES RECOMMENDATIONS\n"
        "------------------------------\n"
        "- Improve marketing efforts\n"
        "- Increase product visibility\n"
        "- Increase product availability\n"
        "- Monitor discounts carefully\n"
        "- Focus on customer satisfaction"
    ),

    "low_sales": (
        "LOW SALES ANALYSIS\n"
        "------------------------------\n"
        "- Sales may be affected by low customer demand\n"
        "- Excessive discounts may reduce profit\n"
        "- Low product visibility may affect sales\n"
        "- Review pricing and marketing strategies\n"
        "- Compare performance across regions and channels"
    ),

    "discount": (
        "DISCOUNT GUIDANCE\n"
        "------------------------------\n"
        "- Discounts can attract more customers\n"
        "- Discounts may increase sales volume\n"
        "- Excessive discounts may reduce profitability\n"
        "- Use discounts strategically\n"
        "- Monitor the effect of discounts on sales"
    ),

    "analysis": (
        "SALES ANALYSIS\n"
        "------------------------------\n"
        "- Monitor sales trends regularly\n"
        "- Compare sales across regions\n"
        "- Evaluate discount effectiveness\n"
        "- Review sales channel performance\n"
        "- Identify high-performing products"
    ),

    "profit": (
        "PROFIT IMPROVEMENT STRATEGIES\n"
        "------------------------------\n"
        "- Reduce unnecessary discounts\n"
        "- Increase sales volume\n"
        "- Focus on high-performing products\n"
        "- Improve pricing strategies\n"
        "- Reduce unnecessary business costs"
    ),

    "inventory": (
        "INVENTORY MANAGEMENT\n"
        "------------------------------\n"
        "- Maintain sufficient stock levels\n"
        "- Monitor fast-moving products\n"
        "- Avoid stock shortages\n"
        "- Avoid overstocking\n"
        "- Review inventory regularly"
    )
}

unrelated_response = (
    "RETAIL ASSISTANT\n"
    "------------------------------\n"
    "I'm not sure what you mean. I can help with retail questions "
    "about sales, forecasting, discounts, profit, analysis and inventory.\n\n"
    "Try asking, for example:\n"
    "- make a sales forecast\n"
    "- how can i increase sales\n"
    "- why are sales low\n"
    "- should i use discounts\n"
    "- how can i improve profit\n"
    "- how do i manage stock"
)

greeting_response = (
    "RETAIL ASSISTANT\n"
    "------------------------------\n"
    "Hello! Ask me about sales, forecasting, discounts, profit, "
    "analysis or inventory.\n\n"
    "For example: make a sales forecast"
)

thanks_response = (
    "RETAIL ASSISTANT\n"
    "------------------------------\n"
    "You're welcome! Ask me another question whenever you are ready."
)


# 5. SALES FORECAST (uses the saved regression model)

def read_number(prompt, minimum=None, maximum=None):

    while True:

        try:
            value = float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if minimum is not None and value < minimum:
            print("Value cannot be less than", minimum)
            continue

        if maximum is not None and value > maximum:
            print("Value cannot be more than", maximum)
            continue

        return value


def read_choice(title, options):

    print("\n" + title)

    keys = list(options.keys())

    for key in keys:
        print(key + ". " + options[key])

    while True:

        choice = input("Choose " + ", ".join(keys) + ": ").strip()

        if choice in options:
            return options[choice]

        print("Invalid choice. Please try again.")


def make_forecast():

    print("\n" + "=" * 60)
    print("              SALES FORECAST")
    print("=" * 60)
    print("Enter the information for the new transaction.")
    print("-" * 60)

    quantity = read_number("Quantity: ", minimum=0)
    unit_price = read_number("Unit price: ", minimum=0)
    discount_pct = read_number(
        "Discount percentage (e.g. 10 for 10%): ",
        minimum=0,
        maximum=100
    )

    transaction_date = pd.to_datetime(
        input("Transaction date (YYYY-MM-DD): "),
        errors="coerce"
    )

    if pd.isna(transaction_date):
        print("\nERROR: Invalid date. Forecast cancelled.")
        return None

    sales_channel = read_choice(
        "Sales Channel",
        {"1": "In-Store", "2": "Online", "3": "Mobile App"}
    )

    payment_method = read_choice(
        "Payment Method",
        {
            "1": "Cash",
            "2": "Credit Card",
            "3": "Debit Card",
            "4": "Gift Card",
            "5": "PayPal"
        }
    )

    region = read_choice(
        "Region",
        {
            "1": "Central",
            "2": "East",
            "3": "North",
            "4": "South",
            "5": "West"
        }
    )

    new_data = pd.DataFrame({
        "quantity": [quantity],
        "unit_price": [unit_price],
        "discount_pct": [discount_pct],
        "year": [transaction_date.year],
        "month": [transaction_date.month],
        "day": [transaction_date.day],
        "day_of_week": [transaction_date.dayofweek],
        "sales_channel": [sales_channel],
        "payment_method": [payment_method],
        "region": [region]
    })

    new_data = pd.get_dummies(
        new_data,
        columns=["sales_channel", "payment_method", "region"],
        drop_first=True,
        dtype=int
    )

    # Same columns, same order as the trained model
    new_data = new_data.reindex(columns=model_features, fill_value=0)

    predicted_sales = sales_model.predict(new_data)[0]

    print("\n" + "=" * 60)
    print("              FORECAST RESULT")
    print("=" * 60)
    print("Quantity:", quantity)
    print("Unit price:", unit_price)
    print("Discount:", discount_pct, "%")
    print("Transaction date:", transaction_date.date())
    print("Sales channel:", sales_channel)
    print("Payment method:", payment_method)
    print("Region:", region)
    print("-" * 60)
    print("PREDICTED SALES AMOUNT: R", round(predicted_sales, 2))
    print("=" * 60)

    return predicted_sales


def describe_prediction(predicted_sales):

    if predicted_sales < 1000:
        level = "LOW"
        advice = (
            "- Review the pricing strategy\n"
            "- Improve marketing and product visibility\n"
            "- Consider targeted promotions"
        )

    elif predicted_sales < 5000:
        level = "MODERATE"
        advice = (
            "- Continue monitoring sales performance\n"
            "- Review discount effectiveness\n"
            "- Identify opportunities to increase sales"
        )

    else:
        level = "HIGH"
        advice = (
            "- Maintain the current sales strategy\n"
            "- Ensure sufficient product availability\n"
            "- Continue effective marketing strategies"
        )

    return (
        "LATEST SALES PREDICTION\n"
        "------------------------------\n"
        "Predicted sales amount: R " + format(predicted_sales, ",.2f") + "\n"
        "Sales level: " + level + "\n\n"
        "Recommendations:\n" + advice
    )


# 6. CHATBOT LOGIC

latest_prediction = None


def correct_spelling(text):

    # Replaces unknown words with the closest known word,
    # for example "forcast" becomes "forecast".

    fixed_words = []

    for word in text.lower().split():

        word = word.strip("?.!,;:")

        if word == "" or word in vocabulary:
            fixed_words.append(word)
            continue

        matches = difflib.get_close_matches(
            word,
            vocabulary,
            n=1,
            cutoff=0.8
        )

        if matches:
            fixed_words.append(matches[0])
        else:
            fixed_words.append(word)

    return " ".join(fixed_words)


def get_intent(user_input):

    cleaned_input = correct_spelling(user_input)

    user_vector = vectorizer.transform([cleaned_input])

    # No known words in the question
    if user_vector.nnz == 0:
        return None

    probabilities = chat_model.predict_proba(user_vector)[0]

    # Not confident that this is a retail topic
    if max(probabilities) < 0.35:
        return None

    return chat_model.classes_[probabilities.argmax()]


def ask_yes_no(question):

    while True:

        answer = input(question + " (yes/no): ").strip().lower()

        if answer in ("yes", "y"):
            return True

        if answer in ("no", "n"):
            return False

        print("Please type yes or no.")


def run_forecast_from_chat():

    global latest_prediction

    result = make_forecast()

    if result is None:
        return "The forecast was not completed."

    latest_prediction = result

    return describe_prediction(latest_prediction)


def chat(user_input):

    global latest_prediction

    words = user_input.lower().replace("!", "").replace("?", "").split()

    # Greetings and thanks
    if words and words[0] in ("hi", "hello", "hey"):
        return greeting_response

    if "thanks" in words or "thank" in words:
        return thanks_response

    intent = get_intent(user_input)

    if intent is None:
        return unrelated_response

    if intent == "prediction":

        if latest_prediction is not None:

            print("\nRetail Assistant:")
            print(describe_prediction(latest_prediction))
            print()

            if ask_yes_no("Would you like to make a new forecast?"):
                return run_forecast_from_chat()

            return "Okay. Ask me another question whenever you are ready."

        print("\nRetail Assistant:")
        print("No sales prediction has been made yet.\n")

        if ask_yes_no("Would you like to make a forecast now?"):
            return run_forecast_from_chat()

        return "Okay. Ask me another question whenever you are ready."

    return responses[intent]


def start_assistant():

    print("\n" + "=" * 60)
    print("              RETAIL ASSISTANT CHATBOT")
    print("=" * 60)

    print("\nHello! Welcome to the Retail Assistant.")
    print(
        "\nI can help you with:"
        "\n- Sales"
        "\n- Sales forecasts and predictions (I can make a forecast for you)"
        "\n- Discounts"
        "\n- Profit"
        "\n- Sales analysis"
        "\n- Inventory"
    )

    print(
        "\nHow to ask: type a full question at the 'You:' prompt."
        "\nExamples:"
        "\n  make a sales forecast"
        "\n  how can i increase sales"
        "\n  why are sales low"
        "\n  should i use discounts"
        "\n  how can i improve profit"
        "\n  how do i manage stock"
    )

    print("=" * 60)

    next_question = None

    while True:

        if next_question is not None:
            user_question = next_question
            next_question = None
            print("\nYou:", user_question)

        else:
            user_question = input("\nYou: ").strip()

        if user_question == "":
            print("Please enter a question.")
            continue

        response = chat(user_question)

        print("\nRetail Assistant:")
        print(response)

        while True:

            print("\n" + "=" * 60)
            print("1. Ask another question")
            print("2. Exit")
            print("(or just type your next question)")
            print("=" * 60)

            choice = input("Choose 1, 2, or type a question: ").strip()

            if choice == "1":
                break

            elif choice == "2":
                print("\nThank you for using the Retail Assistant Chatbot!")
                print("We hope we were able to assist you with your retail business.")
                return

            elif choice == "":
                print("\nPlease choose 1, 2, or type a question.")

            else:
                # The user typed a question instead of a menu number
                next_question = choice
                break


if __name__ == "__main__":
    start_assistant()
