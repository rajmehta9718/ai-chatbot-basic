# AI Product Assistant

A simple Python CLI app that uses the OpenAI API to return structured product information in JSON format.

## Features
- Takes user input from the command line
- Uses an OpenAI model to generate product information
- Returns structured JSON output
- Parses and displays:
  - product_name
  - price
  - category
- Handles invalid JSON and API errors
- Supports continuous interaction with an exit command

## Tech Stack
- Python
- OpenAI API
- python-dotenv

## Setup
1. Create a virtual environment
2. Install dependencies:
   pip install openai python-dotenv
3. Create a `.env` file:
   OPENAI_API_KEY=your_api_key_here
4. Run:
   python app.py

## Example
Input:
suggest a product for students

Output:
- Product Name: Smart Study Lamp
- Price: 29
- Category: Education
