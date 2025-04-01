"""
Example membership inference attack against a deep net classifier on the CIFAR10 dataset
with visualization of successful attack examples
"""
import numpy as np
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import ShadowModelBundle, AttackModelBundle, prepare_attack_data

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000
NUM_EXAMPLES_TO_SHOW = 5  # Number of successful examples to display for each category

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 12, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 6, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 3, "Number of epochs to train attack models.")


def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def display_successful_examples(original_images, predictions, actual_labels, attack_guesses, real_membership, cifar10_class_names=None):
    """Display examples where the membership inference attack was successful."""
    if cifar10_class_names is None:
        cifar10_class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    # Define success types to display
    success_types = [
        ("True Positives", np.logical_and(attack_guesses == 1, real_membership == 1)),  # Correctly identified as member
        ("True Negatives", np.logical_and(attack_guesses == 0, real_membership == 0)),  # Correctly identified as non-member
        ("False Positives", np.logical_and(attack_guesses == 1, real_membership == 0)),  # Incorrectly identified as member
        ("False Negatives", np.logical_and(attack_guesses == 0, real_membership == 1))   # Incorrectly identified as non-member
    ]
    
    # For each success type, show examples
    for name, condition in success_types:
        indices = np.where(condition)[0]
        
        if len(indices) == 0:
            print(f"No {name} examples found.")
            continue
            
        # Select up to NUM_EXAMPLES_TO_SHOW examples
        num_examples = min(NUM_EXAMPLES_TO_SHOW, len(indices))
        selected_indices = indices[:num_examples]
        
        fig, axes = plt.subplots(1, num_examples, figsize=(3*num_examples, 4))
        fig.suptitle(f"{name} Examples", fontsize=16)
        
        if num_examples == 1:
            axes = [axes]  # Make iterable for single example case
            
        for i, idx in enumerate(selected_indices):
            # Display the image
            axes[i].imshow(np.clip(original_images[idx], 0, 1))
            
            # Get the true class
            true_class = np.argmax(actual_labels[idx])
            class_name = cifar10_class_names[true_class]
            
            # Get the model's prediction confidences
            pred_confidences = predictions[idx]
            pred_class = np.argmax(pred_confidences)
            conf = pred_confidences[pred_class]
            
            # Set the title with prediction info
            title = f"Class: {class_name}\n"
            title += f"Pred: {cifar10_class_names[pred_class]} ({conf:.2f})\n"
            
            # Add membership info
            if real_membership[idx] == 1:
                title += "Actual: In Training Set\n"
            else:
                title += "Actual: Not In Training Set\n"
                
            if attack_guesses[idx] == 1:
                title += "MIA: Predicted In Training"
            else:
                title += "MIA: Predicted Not In Training"
                
            axes[i].set_title(title, fontsize=9)
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()


def demo(argv):
    del argv 

    (X_train, y_train), (X_test, y_test) = get_data()

    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        X_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
    )

    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )

    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

    # Prepare data for attack evaluation
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Merge the original images for visualization
    original_images_in = X_train[:ATTACK_TEST_DATASET_SIZE] 
    original_images_out = X_test[:ATTACK_TEST_DATASET_SIZE]
    original_images = np.vstack([original_images_in, original_images_out])
    
    # Get the predictions from target model
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )
    
    # Store actual labels
    actual_labels = np.vstack([y_train[:ATTACK_TEST_DATASET_SIZE], 
                              y_test[:ATTACK_TEST_DATASET_SIZE]])

    # Get attack predictions
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)
    attack_precision = np.sum((attack_guesses == 1) & (real_membership_labels == 1)) / np.sum(attack_guesses == 1)
    attack_recall = np.sum((attack_guesses == 1) & (real_membership_labels == 1)) / np.sum(real_membership_labels == 1)
    
    print("Attack accuracy:", attack_accuracy)
    print("Attack precision:", attack_precision)
    print("Attack recall:", attack_recall)
    
    # Display detailed metrics
    print("\nDetailed metrics:")
    print(f"True Positives: {np.sum((attack_guesses == 1) & (real_membership_labels == 1))}")
    print(f"True Negatives: {np.sum((attack_guesses == 0) & (real_membership_labels == 0))}")
    print(f"False Positives: {np.sum((attack_guesses == 1) & (real_membership_labels == 0))}")
    print(f"False Negatives: {np.sum((attack_guesses == 0) & (real_membership_labels == 1))}")
    
    # Calculate per-class attack success rate
    print("\nPer-class attack success rate:")
    for class_idx in range(NUM_CLASSES):
        class_indices = np.argmax(actual_labels, axis=1) == class_idx
        if np.sum(class_indices) > 0:
            class_accuracy = np.mean(attack_guesses[class_indices] == real_membership_labels[class_indices])
            print(f"Class {class_idx}: {class_accuracy:.4f}")
            
    # Display successful examples
    print("\nDisplaying examples of successful membership inference attacks...")
    display_successful_examples(
        original_images,
        attack_test_data,  # These are the target model's predictions
        actual_labels,
        attack_guesses,
        real_membership_labels
    )


if __name__ == "__main__":
    app.run(demo)
