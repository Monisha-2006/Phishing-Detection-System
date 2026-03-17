import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set(style="darkgrid")

# -----------------------
# URL DATASET
# -----------------------

url_df = pd.read_excel("URL_Detection/phishing_url_dataset_unique.xlsx")

# convert label to numeric if needed
url_df['label'] = url_df['label'].astype(int)

# URL length
url_df['url_length'] = url_df['url'].apply(len)

# Chart 1 (existing)
plt.figure(figsize=(6,4))
sns.countplot(x=url_df['label'], palette="Set2")
plt.title("Phishing vs Legitimate URLs")
plt.savefig("static/url_count.png")
plt.close()

# Chart 2 (existing)
plt.figure(figsize=(6,4))
sns.histplot(url_df['url_length'], bins=40)
plt.title("URL Length Distribution")
plt.savefig("static/url_length.png")
plt.close()

# -----------------------
# NEW CHART 1 — Pie Chart
# -----------------------

plt.figure(figsize=(6,6))
url_df['label'].value_counts().plot.pie(
autopct='%1.1f%%',
colors=['skyblue','salmon']
)
plt.title("Phishing vs Legitimate URLs")
plt.ylabel("")
plt.savefig("static/url_pie.png")
plt.close()

# -----------------------
# NEW CHART 2 — Boxplot
# -----------------------

plt.figure(figsize=(6,4))
sns.boxplot(x=url_df['label'], y=url_df['url_length'], palette="coolwarm")
plt.title("URL Length vs Label")
plt.savefig("static/url_box.png")
plt.close()

# -----------------------
# NEW CHART 3 — Dot count
# -----------------------

url_df['dot_count'] = url_df['url'].str.count(r'\.')

plt.figure(figsize=(6,4))
sns.histplot(url_df['dot_count'], bins=20)
plt.title("Dot Count Distribution")
plt.savefig("static/url_dots.png")
plt.close()

# -----------------------
# NEW CHART 4 — Heatmap
# -----------------------

corr = url_df[['label','url_length','dot_count']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("static/url_heatmap.png")
plt.close()

print("URL charts created")
# ----------------------------
# SMS DATASET ANALYSIS
# ----------------------------

sms_df = pd.read_csv("SMS_Detection/sms_data.csv")
sms_df['LABEL'] = sms_df['LABEL'].map({'ham':0,'spam':1})
sms_df['length'] = sms_df['TEXT'].apply(len)

# Pie chart
plt.figure(figsize=(6,6))
sms_df['LABEL'].value_counts().plot.pie(
autopct='%1.1f%%',
colors=['lightgreen','orange']
)
plt.title("Spam vs Legitimate SMS")
plt.ylabel("")
plt.savefig("static/sms_pie.png")
plt.close()

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(sms_df['length'], bins=40)
plt.title("SMS Length Distribution")
plt.savefig("static/sms_length.png")
plt.close()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=sms_df['LABEL'], y=sms_df['length'])
plt.title("SMS Length vs Label")
plt.savefig("static/sms_box.png")
plt.close()

# Heatmap
corr = sms_df[['LABEL','length']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("SMS Feature Correlation")
plt.savefig("static/sms_heatmap.png")
plt.close()

# -----------------------
# SMS WORD CLOUD
# -----------------------

sms_text = " ".join(sms_df['TEXT'].astype(str))

wordcloud = WordCloud(
width=800,
height=400,
background_color='white',
colormap='plasma'
).generate(sms_text)

plt.figure(figsize=(8,4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("SMS WordCloud")
plt.savefig("static/sms_wordcloud.png")
plt.close()

print("SMS charts created")
# ----------------------------
# EMAIL DATASET ANALYSIS
# ----------------------------

email_df = pd.read_csv("Email_Detection/email_data.csv")

email_df['length'] = email_df['text_combined'].apply(len)

# Pie chart
plt.figure(figsize=(6,6))
email_df['label'].value_counts().plot.pie(
autopct='%1.1f%%',
colors=['lightblue','red']
)
plt.title("Phishing vs Legitimate Emails")
plt.ylabel("")
plt.savefig("static/email_pie.png")
plt.close()

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(email_df['length'], bins=40)
plt.title("Email Length Distribution")
plt.savefig("static/email_length.png")
plt.close()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=email_df['label'], y=email_df['length'])
plt.title("Email Length vs Label")
plt.savefig("static/email_box.png")
plt.close()

# Heatmap
corr = email_df[['label','length']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Email Feature Correlation")
plt.savefig("static/email_heatmap.png")
plt.close()

# -----------------------
# EMAIL WORD CLOUD
# -----------------------

email_text = " ".join(email_df['text_combined'].astype(str))

wordcloud = WordCloud(
width=800,
height=400,
background_color='white',
colormap='cool'
).generate(email_text)

plt.figure(figsize=(8,4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Email WordCloud")
plt.savefig("static/email_wordcloud.png")
plt.close()

print("EMAIL charts created")
print("Analysis charts generated successfully!")