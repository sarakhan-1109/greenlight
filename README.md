# Greenlight

Greenlight is a web application that estimates a film's likely commercial
outcome before it is released. A user enters basic information about a planned
film, such as its budget, genre, lead actor, release month, runtime, and whether
it is part of an existing franchise. The application returns a predicted
box-office tier (Flop, Moderate, Hit, or Blockbuster), a breakdown of which
inputs most influenced the prediction, and a short written explanation of the
result.

- Live application: https://greenlight-jet.vercel.app
- Source code: https://github.com/sarakhan-1109/greenlight

## Overview

Studios decide whether to fund a film, known in
the industry as "greenlighting" it, well before the film is made and with
limited information. Greenlight looks at the information available at that early
stage and estimates how a film is likely to perform, based on patterns in past
films.

The aim of the project was not to build a perfect predictor. Box-office
performance is genuinely hard to predict. The aim was to build a clear and honest
system and to avoid the common mistakes that make a model look accurate in
testing but fail in practice.

## How it works

```
User (web browser)
        |
        v
React single-page application  (hosted on Vercel)
        |
        |  film details: budget, genre, lead actor, timing, runtime, franchise
        v
FastAPI backend service  (hosted on Render)
        |
        |--> Machine learning model  ->  predicted tier + factor contributions
        |
        |--> Language model (Gemini) ->  plain-English explanation of the result
        |
        v
Result sent back to the page
```

The user fills in a single web page. When they submit a film's details, the page
sends them to a backend service. The backend does two things. First, a trained
machine learning model predicts the tier and reports how much each input
contributed to that prediction. Second, a language model takes the finished
prediction and writes a short explanation of it in plain English. Both are
returned to the page and displayed.

One point is central to the design: the machine learning model makes the
prediction, and the language model only describes the result in words. The
language model never decides the outcome. The two roles are kept in separate
parts of the code.

## The data

The model is trained on a public dataset of about 7,700 films released between
1980 and 2020. Each film record includes its budget, genre, release date,
runtime, lead actor, production company, and gross revenue, among other fields.

Real data of this kind is messy, and handling that was part of the work:

- About 28 percent of the films had no recorded budget. Because budget is the
  single most useful input, these films were removed rather than filled in with
  a guessed value. Guessing the most important input would have added noise. This
  left 5,430 films with complete information.
- Some genres appeared only a few times. These rare genres were grouped into a
  single "Other" category so they would not make the model unstable.
- Two fields, the audience score and the number of votes, were removed
  completely. Both are only known after a film is released. This is explained in
  the next section.

## Avoiding data leakage

The most important decision in this project was to use only information that
would be known before a film is released.

Data leakage is a common and serious mistake in machine learning. It happens when
a model is accidentally given information that would not really be available at
the moment a prediction is needed. A model built this way scores very well in
testing but is useless in practice, because it depends on information you do not
actually have.

The original dataset included several fields describing what happened after
release, such as the audience score and the number of votes. These are closely
related to box-office success, so including them would have made the model look
very accurate. But they cannot be known before release, so using them would have
been meaningless. They were removed.

The gross revenue field was kept for one purpose only: to define the four tiers
the model predicts. It is never used as an input.

This rule also required care when measuring an actor's star power. A simple
approach would be to use an actor's overall popularity, but much of that
popularity is built up by the very films being studied. Instead, for each film,
star power is measured only from the actor's earlier films, using the average
revenue of the films they appeared in before that film's release date. This
information would genuinely be available at the time and does not reveal future
results.

## The model

The system predicts one of four tiers rather than an exact dollar amount.
Predicting an exact figure would imply a level of precision the data does not
support. Four tiers are more honest and easier to read.

The tiers are defined by sorting all films by gross revenue and dividing them
into four equal groups:

| Tier | Box-office revenue |
| --- | --- |
| Flop | under about $10.7 million |
| Moderate | about $10.7 to $36.8 million |
| Hit | about $36.8 to $112 million |
| Blockbuster | over about $112 million |

Because the groups are equal in size, each tier contains a quarter of all films.
This means a model that simply guessed at random would be correct about 25
percent of the time, which gives a clear baseline to measure against.

The inputs used by the model are all available before release:

- Production budget
- Genre
- Lead actor star power, expressed as a 1 to 5 level based on the actor's
  earlier films
- Release month
- Whether the film is part of a franchise or a sequel, identified from its title
- Runtime

The model is XGBoost, a standard and widely used method that combines many small
decision trees. It was chosen because it works well on this kind of tabular data
and because its decisions can be explained. A deeper version of the model was
tested and found to overfit, meaning it memorized the training data and did worse
on new films, so a simpler and shallower version was kept.

## Explaining the prediction

The application shows two kinds of explanation.

The first is a chart of contributing factors. For each prediction, the model
reports how much each input pushed the result toward or away from the predicted
tier. This uses a standard technique built into the model, so the explanation
comes directly from the model itself rather than from a separate guess.

The second is a short written note. The model's output, meaning the tier, the
confidence level, and the factor contributions, is passed to a language model
(Google Gemini), which writes two or three sentences describing the result in
plain language. The language model is given the finished prediction and asked
only to explain it. If the language model is unavailable, the application falls
back to a fixed template so that it always returns a readable explanation.

## Results

Box-office prediction is difficult, and the results reflect that honestly.

- On a standard test set, the model predicts the exact tier about 50 percent of
  the time, and either the correct tier or an adjacent one about 90 percent of
  the time. Adjacent means, for example, predicting "Hit" when the true tier is
  "Moderate."
- Five-fold cross-validation, which repeats the training and testing five times
  on different splits of the data, gives 52.2 percent exact accuracy with very
  little variation (about plus or minus 1 percent). This shows the result is
  stable and not the product of a single lucky split.
- A stricter test, training only on films released up to 2012 and testing on
  films from 2013 onward, gives 56.4 percent exact accuracy. This test is more
  realistic than a random split because it asks the model to predict films from a
  later period than it was trained on. The model held up, which suggests it
  generalizes to films it has not seen.

The model is most accurate at the extremes, meaning clear flops and clear
blockbusters, and least accurate in the middle tiers. This matches how
box-office performance actually behaves, where large productions and very small
productions are more predictable than films in between.

An accuracy near 50 percent is a reasonable and honest result for this problem.
If the model had scored much higher, for example above 90 percent, that would
more likely point to a data leakage problem than to genuine skill. Keeping the
result realistic was a deliberate choice.

## Technology used

| Part of the project | Technology |
| --- | --- |
| Data preparation and model | Python, pandas, XGBoost, scikit-learn |
| Backend service | FastAPI |
| Frontend | React (built with Vite) |
| Explanation layer | Google Gemini |
| Hosting | Vercel (frontend), Render (backend) |

The application does not use a database and does not store user data. Each
prediction is handled on its own.

## Project structure

```
backend/    FastAPI service, the trained model, and the prediction logic
frontend/   React single-page application
model/      Scripts for cleaning the data, training the model, and validating it
```

## Running the project locally

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

Frontend, in a separate terminal:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

The written explanation is optional. Without an API key the application uses a
built-in template instead. To enable the language model, add a free Google Gemini
key (available from Google AI Studio, no payment required) to `backend/.env`.

## Limitations and possible improvements

- The model predicts gross revenue tiers, not profit. A film can earn a large
  gross and still lose money relative to its budget. A version that predicts
  profitability would be a useful extension.
- Budgets and revenues are not adjusted for inflation. The model is given the
  release year to partly account for this, but a proper inflation adjustment
  would be more accurate.
- Franchise status is inferred from the film's title, which is an approximation
  rather than a precise record.
- The dataset ends in 2020. Testing on more recent films would show how well the
  model holds up on the current market.

Further extensions could include a profitability target, an option to compare two
films side by side, and a larger set of inputs.
