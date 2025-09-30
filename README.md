# 🎓 Scholarship Recommender App

The **Scholarship Recommender App** is a Streamlit-based tool that helps students and applicants discover scholarships that best match their profiles.  

It leverages **Natural Language Processing (NLP)** techniques to compare user input with available scholarship descriptions and ranks them by relevance.  

This project aims to simplify the scholarship search process, reduce manual effort, and guide applicants towards opportunities aligned with their goals.

## ⚙️ Features

- Upload and process scholarships from an **Excel file (`scholarships.xlsx`)**.  
- Clean and normalize scholarship data:
  - Remove duplicates
  - Parse deadlines
  - Preprocess text for analysis
- Recommend scholarships based on:
  - Level of study
  - Field of study
  - Country of study
  - Text similarity (TF-IDF + cosine similarity)
- Display **top-N recommendations** with scores, deadlines, and official links.  
- Option to **download recommendations as CSV**.

## 🖼️ Demo Preview
![Scholarship Recommender Demo](https://github.com/user-attachments/assets/c77788d5-5025-4ebb-a158-c56acdcfd695)

Try it live here:  
[Scholarship Recommender Prototype](https://scholarship-recommender.streamlit.app/)

## 🚀 Quick Start (Local Setup)

**Clone the repo:**

```bash
git clone https://github.com/Hajaratitilope/scholarship-recommender.git
cd scholarship-recommender
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the app:**

```bash
streamlit run app.py
```

> Note: The app uses your local scholarships.xlsx file by default. You can also upload any other .xlsx file in the same format.

## 📂 Dataset

This app works with an **Excel file (`scholarships.xlsx`)** containing scholarship information.  

- The file should include columns such as:
  - Scholarship Name  
  - Host/Donor  
  - Type of Funding  
  - Eligibility  
  - Benefits  
  - Application Deadline  
  - Official Source Link  

- You can either:
  - Upload your own Excel file through the Streamlit interface, or  
  - Place a local `scholarships.xlsx` file in the app folder.  

⚠️ **Note:** For demo purposes, this repo includes a `scholarships.xlsx` file. You can replace it with your own dataset as long as it follows the same column structure.

## 🔧 How to Contribute / Extend

You can extend or customize this recommender in several ways:

- **Add new scholarships** by updating your local `scholarships.xlsx` file.  
- **Tune the recommendation** by adjusting the `alpha` slider in the UI (balances text similarity vs. rule-based matching).  
- **Enhance the rules** by editing the `recommend()` function in `app.py` to capture more criteria (e.g., region, GPA, age limits).  
- **Improve the UI** with Streamlit widgets for better interactivity.  
- **Deploy your own app** on [Streamlit Cloud](https://streamlit.io/cloud) or other platforms.  

Feel free to fork this repo, suggest improvements, or submit a pull request.

## 🙏 Acknowledgements  

This project is guided by **Careergrill** (CEO and Founder: *Sanni Alausa*)  
🌐 Website: [careergrill.com](https://www.careergrill.com/)  

Developed by **Hajarat Titilope Olufade**  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/olufade-hajarat-726156180) as part of an ongoing portfolio in **AI/ML applications**.  



