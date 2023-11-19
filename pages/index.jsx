import styles from "./index.module.scss";
import { useState } from "react";
import Modal from "./components/Modal.tsx";
import MainModal from "./components/MainModal.tsx";

const IssuesPage = () => {
  const [modal, setModal] = useState(false);
  const [tweet, setTweet] = useState(null);
  const [polarity, setPolarity] = useState(null);
  const [pol, setPol] = useState(null);
  const [mainModal, setMainModal] = useState(false);
  const [modal2, setModal2] = useState(false);
  const [emotion1, setEmotion1] = useState(null);
  const [emotion2, setEmotion2] = useState(null);
  const [emotion3, setEmotion3] = useState(null);
  const [emotion, setEmotion] = useState(0);
  const [modal3, setModal3] = useState(false);
  const [intensityText, setIntensityText] = useState(false);
  const [modal4, setModal4] = useState(false);
  const [resultExplore, setResultExplore] = useState(null);
  const [resultSudden, setResultSudden] = useState(null);
  const [text, setText] = useState(false);
  const [modalText, setModalText] = useState(false);
  const [deepen, setDeepen] = useState(null);
  const [frequent_word, setFW] = useState(null);

  const [formData, setFormData] = useState({
    tweetText: "",
    tweetIntensity: "",
    tweetFrequency: "",
  });

  const closeModal = () => {
    setModal(false);
    setModal2(false);
    setModal3(false);
  };

  const closeModal4 = () => {
    setModal4(false);
    setResultExplore(null);
  };

  const closeMainModal = () => {
    setMainModal(false);
    setTweet("");
    setPolarity("");
    setPol("");
    setFW("");
    setText(false);
  };

  const handleApiCall = async (url, options, successCallback) => {
    try {
      const response = await fetch(url, options);

      if (response.ok) {
        const result = await response.json();
        successCallback(result);
      } else {
        console.error("Failed to send data:", response.statusText);
      }
    } catch (error) {
      console.error("Error sending data:", error);
    }
  };

  const handleSubmitTweet = async (event) => {
    event.preventDefault();

    const url = "http://127.0.0.1:5000/edit_tweet";
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ tweet: formData.tweetText }),
    };

    handleApiCall(url, options, (result) => {
      setTweet(result.tweet);
      setPolarity(result.polarity);
      setPol(result.pol);
      setFW(result.frequent_word);
      setFormData({ ...formData, tweetText: "" });
    });
  };

  const handleSubmitDeep = async (deep) => {
    const url = "http://127.0.0.1:5000/deepen_emotion";
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ deep: deep }),
    };

    handleApiCall(url, options, (result) => {
      setDeepen(result.deepen);
    });
  };

  const handleSubmitEmotion = async (modal4Choice) => {
    const url = "http://127.0.0.1:5000/explore_emotion";
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ t_op: emotion, task: modal4Choice }),
    };

    handleApiCall(url, options, (result) => {
      setResultExplore(result.result);
    });
  };

  const fetchData = async () => {
    const url = "http://127.0.0.1:5000/get_emotions";

    handleApiCall(url, {}, (result) => {
      setEmotion1(result.emotion1);
      setEmotion2(result.emotion2);
      setEmotion3(result.emotion3);
    });
  };

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmitIntensity = async (event) => {
    event.preventDefault();

    const url = "http://127.0.0.1:5000/suddenness";
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        intensity: formData.tweetIntensity,
        frequency: formData.tweetFrequency,
      }),
    };

    handleApiCall(url, options, (result) => {
      setResultSudden(result.result);
      setFormData({ ...formData, tweetIntensity: "", tweetFrequency: "" });
    });
  };

  const handleIntensityInputChange = (event) => {
    const { name, value } = event.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleFrequencyInputChange = (event) => {
    const { name, value } = event.target;
    setFormData({ ...formData, [name]: value });
  };

  return (
    <div>
      <div className={styles.mainContent}>
        <h1>Hi I'm MoodBot!</h1>
        <p>Your virtual agent for tweet recognition</p>
        <img className={styles.imagen} src="/icons/Diseno_sin_titulo2.png" />
        <button
          className={styles.startButton}
          onClick={() => setMainModal(true)}
        >
          Start
        </button>
      </div>

      <MainModal isOpen={mainModal} onClose={closeMainModal}>
        <h1>Write your tweet</h1>
        <form onSubmit={handleSubmitTweet}>
          <label htmlFor="tweet" className={styles.inp}>
            <textarea
              id="tweet"
              placeholder=" "
              name="tweetText"
              value={formData.tweetText}
              onChange={handleInputChange}
            />
            <span className={styles.label}>Tweet</span>
            <span className={styles["focus-bg"]}></span>
          </label>
          <div className={styles.tweetButtonContainer}>
            <button
              onClick={() => setText(true)}
              className={styles.tweetButton}
              type="submit"
            >
              Submit Tweet
            </button>
          </div>
        </form>
        {!text && (
          <button className={styles.closeButton} onClick={closeMainModal}>
            Exit
          </button>
        )}
        {text && (
          <>
            <p className={styles.mainText}>
              You tweeted: {tweet}.
              <br />
              Ohh wow the polarity is {polarity} that seems like a {pol}{" "}
              emotion.
              <br />
              The most frequent word is {frequent_word}.
              <br />
              What do you want to do?
            </p>
            <div className={styles.optionsContainer}>
              <button
                className={styles.optionButton}
                onClick={() => setModal(true)}
              >
                Go deeper
              </button>
              <button
                className={styles.optionButton}
                onClick={() => {
                  setModal2(true);
                  fetchData();
                }}
              >
                Explore emotion
              </button>
              <button
                className={styles.optionButton}
                onClick={() => setModal3(true)}
              >
                Know suddenness
              </button>
              <button className={styles.optionButton} onClick={closeMainModal}>
                Exit
              </button>
            </div>
          </>
        )}
      </MainModal>

      <Modal isOpen={modal} onClose={closeModal}>
        <h1>Go deeper</h1>
        <p className={styles.modalText}>
          From 1 to 3, how specific do you want to know about your emotion in
          this tweet?
        </p>
        <div className={styles.modal.buttonContainer}>
          {[1, 2, 3].map((level) => (
            <button
              key={level}
              className={styles.optionButton}
              onClick={() => {
                handleSubmitDeep(level);
                setModalText(true);
              }}
            >
              {level}
            </button>
          ))}
        </div>

        {modalText && (
          <p className={styles.modalText}>
            According to my analysis {deepen} is a more specific emotion related
            to your tweet
          </p>
        )}
        <button
          className={styles.closeButton}
          onClick={() => {
            closeModal();
            setModalText(false);
          }}
        >
          Close Modal
        </button>
      </Modal>

      <Modal isOpen={modal2} onClose={closeModal}>
        <h1>Explore emotion</h1>
        <p className={styles.modalText}>
          Which of these words do you identify with the most?
        </p>
        <div className={styles.modal.buttonContainer}>
          {[emotion1, emotion2, emotion3].map((emotion, index) => (
            <button
              key={index}
              className={styles.optionButton}
              onClick={() => {
                setEmotion(index + 1);
                setModal4(true);
              }}
            >
              {emotion}
            </button>
          ))}
        </div>

        <button className={styles.closeButton} onClick={closeModal}>
          Close Modal
        </button>
      </Modal>
      <Modal isOpen={modal4} onClose={closeModal4}>
        <h1>Wow, that's nice! What do you want to know?</h1>
        {[1, 2, 3].map((choice) => (
          <>
            <button
              key={choice}
              className={styles.optionButton}
              onClick={() => handleSubmitEmotion(choice)}
            >
              {choice === 1
                ? "Get an emotion related to the word"
                : choice === 2
                ? "Get the polarity of that word"
                : "Get a music recommendation"}
            </button>
            <br />
            <br />
          </>
        ))}
        <p>{resultExplore}</p>
        <button className={styles.closeButton} onClick={closeModal4}>
          Close Modal
        </button>
      </Modal>
      <Modal isOpen={modal3} onClose={closeModal}>
        <h1>Suddenness</h1>
        <p>Ummm... Let me know something</p>
        <p>In a scale from 1 to 10:</p>
        <p>
          How much was the intensity and frequency of the emotion in the tweet?
        </p>
        <form className={styles.intForm} onSubmit={handleSubmitIntensity}>
          <div className={styles.inputContainer}>
            {["Intensity", "Frequency"].map((label, index) => (
              <label key={index} htmlFor={label} className={styles.inpu}>
                <input
                  type="text"
                  name={`tweet${label}`}
                  id={label}
                  placeholder="&nbsp;"
                  value={formData[`tweet${label}`]}
                  onChange={
                    label === "Intensity"
                      ? handleIntensityInputChange
                      : handleFrequencyInputChange
                  }
                />
                <span className={styles.label}>{label}</span>
                <span className={styles["focus-bg"]}></span>
              </label>
            ))}
          </div>
          <div className={styles.intensityButtonContainer}>
            <button
              onClick={() => setIntensityText(true)}
              className={styles.tweetButton}
              type="submit"
            >
              Submit intensity
            </button>
          </div>
        </form>
        {intensityText && (
          <p>
            I can calculate that the suddenness of your emotion was:{" "}
            {resultSudden}
          </p>
        )}

        <button
          className={styles.closeIntensityButton}
          onClick={() => {
            closeModal();
            setIntensityText(false);
          }}
        >
          Close Modal
        </button>
      </Modal>
    </div>
  );
};

export default IssuesPage;
