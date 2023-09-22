import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load your pre-trained model
model = tf.keras.models.load_model('best_model.h5')  # Replace with your model path

# Set page title and description
st.set_page_config(
    page_title="Crop Disease Identification",
    page_icon="🌾",
    layout="wide",
)

# Add a background image or color

page_bg_img = '''
<style>
body {
background-image: url("./11.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


# Set the title and description
st.title("Crop Disease Identification")
st.write("Upload an image of a crop leaf, and we'll identify the disease!")

# Add a sidebar for additional information
st.sidebar.header("About Crop Diseases")
st.sidebar.markdown(
    """
    Crop diseases pose a significant threat to global agriculture, impacting crop yield, food security, and economic stability. These diseases are caused by various pathogens, including fungi, bacteria, viruses, and nematodes, and they target a wide range of crops such as wheat, rice, maize, and potatoes. Crop diseases manifest through visible symptoms like wilting, discoloration, lesions, and stunted growth, making early detection crucial.

The consequences of untreated crop diseases are dire, leading to yield losses, decreased food availability, and increased production costs due to the need for pesticides and fungicides. Additionally, the spread of resistant strains of pathogens further complicates disease management.

Sustainable agriculture practices and innovative technologies are critical in combating crop diseases. Integrated disease management strategies encompass practices like crop rotation, planting disease-resistant varieties, and adopting biological control methods. Furthermore, modern technologies such as remote sensing, image recognition, and artificial intelligence play pivotal roles in early disease detection and monitoring. By identifying diseases promptly and recommending appropriate treatments, these technologies empower farmers to make informed decisions, reduce crop losses, and work towards global food security in an ever-changing agricultural landscape.
    """
)

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values

    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)

    # Define class labels (modify this based on your class labels)
    class_labels = {
 0:'Disease - Apple_scab',
   1:'Disease - Apple Black rot',
     2:'Disease - Cedar apple rust',
     3:'Healthy Apple',
     4:'Healthy Cherry',
     5:'Disease - Cherry Powdery mildew',
    6:'Disease - Corn_Cercospora_leaf_spot Gray_leaf_spot',
    7:'Disease - Corn Common rust',
    8:'Healthy Corn',
    9:'Disease - Grape Black rot',
    10:"Healthy Grape",
    11:"Disease - Grape Leaf blight_(Isariopsis_Leaf_Spot)",
    12:"Disease - Pepper_bell___Bacterial_spot",
    13:"Healthy Pepper",
    14:"Disease - Potato_Early_blight",
    15:"Healthy Potato",
    16:"Disease - Potato_Late_blight"
}

    # Get the predicted class label
    predicted_label = class_labels[class_index[0]]

    # Display the result
    st.subheader("Disease Prediction:")
    st.write(f"The leaf in the image is most likely affected by: **{predicted_label}**")

    # Add a button to show more details
    if st.button("Show More Details"):
        # Provide disease description or additional information
        disease_descriptions = {
            'Disease - Apple_scab': "To know more about the disease, its causes, preventions and cure.\n Click on this link: https://extension.umn.edu/plant-diseases/apple-scab",
            'Disease - Apple Black rot': "To know more about the disease, its causes, preventions and cure.\n Click on this link: https://extension.umn.edu/plant-diseases/black-rot-apple",
            'Disease - Cedar apple rust': "To know more about the disease, its causes, preventions and cure.\n Click on this link: https://kb.jniplants.com/preventing-cedar-apple-rust/#:~:text=While%20removal%20of%20Eastern%20Redcedars,only%20be%20preventative%20not%20curative.",
            'Healthy Apple':"The image is of a healthy Apple tree. ",
            'Healthy Cherry':"The image is of a healthy Cherry tree. ",
            'Disease - Cherry Powdery mildew':"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/",
            'Disease - Corn_Cercospora_leaf_spot Gray_leaf_spot':"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://en.wikipedia.org/wiki/Corn_grey_leaf_spot",
            'Disease - Corn Common rust':"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://extension.umn.edu/corn-pest-management/common-rust-corn",
            'Healthy Corn':"The image is of a healthy Corn tree. ",
            'Disease - Grape Black rot':"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems/diseases/fruit-spots/black-rot-of-grapes#:~:text=Black%20rot%2C%20caused%20by%20the,effect%20is%20to%20the%20fruit.",
            "Healthy Grape":"The image is of a healthy Grape tree. ",
            "Disease - Grape Leaf blight_(Isariopsis_Leaf_Spot)":"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://www.sciencedirect.com/science/article/abs/pii/S0261219414001598",
            "Disease - Pepper_bell___Bacterial_spot":"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/bacterial-leaf-spot-of-pepper",
            "Healthy Pepper":"The image is of a healthy Pepper tree. ",
            "Disease - Potato_Early_blight":"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://ipm.ucanr.edu/agriculture/potato/early-blight/#:~:text=Early%20blight%20is%20primarily%20a,produce%20characteristic%20target%2Dboard%20effect.",
            "Healthy Potato":"The image is of a healthy Potato tree. ",
            "Disease - Potato_Late_blight":"To know more about the disease, its causes, preventions and cure.\n Click on this link: https://www.pau.edu/potato/lb_disease.php#:~:text=Late%20blight%20caused%20by%20the,as%2080%25%20in%20epidemic%20years.",



        }
        disease_description = disease_descriptions.get(predicted_label, "No description available.")
        st.write(disease_description)

    # Add an option to show the probability distribution
    show_probabilities = st.checkbox("Show Probabilities")
    if show_probabilities:
        st.subheader("Class Probabilities:")
        for i, label in enumerate(class_labels.values()):
            probability = prediction[0][i]
            st.write(f"{label}: {probability:.2f}")

    # Visualize the class probabilities as a bar chart
    if show_probabilities:
        probabilities = prediction[0]
        plt.figure(figsize=(10, 6))
        plt.bar(class_labels.values(), probabilities)
        plt.xlabel("Crop Disease Classes")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Probability")
        plt.title("Class Probabilities")
        st.pyplot(plt)

# Add some closing remarks or additional content
st.markdown("For more information or assistance, please contact us on the email : diseasedetection@gamil.com")
