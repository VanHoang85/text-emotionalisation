
def templates(emotion: str, utt_info: dict):
    input_text = utt_info["input"]
    prev_utt = utt_info["context"]
    candidates = {"target": utt_info["reference"].split(":")[0],
                  "wst": utt_info["pred_pegasus_wst"].split(":")[0],
                  "nst": utt_info["pred_pegasus_nst"].split(":")[0],
                  "nps": utt_info["pred_pegasus_np"].split(":")[0],
                  "naive": utt_info["pred_naive"].split(":")[0]}

    emotion_temp = f"""
    <!-- You must include this JavaScript file -->
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

    <!-- For the full list of available Crowd HTML Elements and their input/output documentation,
          please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

    <!-- You must include crowd-form so that your task submits answers to MTurk -->
    <crowd-form answer-format="flatten-objects">

        <p>
          <strong>
            Use the sliders below to answer the questions below.
          </strong>
        </p>

        <p>Q: <strong>How {emotion} is the speaker in each sentence below?</strong> (1 = Not at all, 5 = Very much)</p>
      <p>
        <ul>
          <li>
            <p>{candidates["target"]}</p>
            <p><crowd-slider name="emotion-target" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["wst"]}</p>
            <p><crowd-slider name="emotion-wst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nst"]}</p>
            <p><crowd-slider name="emotion-nst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nps"]}</p>
            <p><crowd-slider name="emotion-nps" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["naive"]}</p>
            <p><crowd-slider name="emotion-naive" min="1" max="5" required pin></crowd-slider></p>
          </li>
        <ul>
      </p>
    </crowd-form>
    """

    similarity_temp = f"""
    <!-- You must include this JavaScript file -->
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

    <!-- For the full list of available Crowd HTML Elements and their input/output documentation,
          please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

    <!-- You must include crowd-form so that your task submits answers to MTurk -->
    <crowd-form answer-format="flatten-objects">

        <p>
          <strong>
            Use the sliders below to answer the question below.
          </strong>
        </p>

      <p>Q: <strong>How close in meaning is each sentence below to the sentence by speaker A?</strong> (1 = Not at all, 5 = Very much)</p>
      <p>
          <p>Speaker A: <strong>{input_text}</strong></p>
        <ul>
          <li>
            <p>{candidates["target"]}</p>
            <p><crowd-slider name="similarity-target" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["wst"]}</p>
            <p><crowd-slider name="similarity-wst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nst"]}</p>
            <p><crowd-slider name="similarity-nst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nps"]}</p>
            <p><crowd-slider name="similarity-nps" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["naive"]}</p>
            <p><crowd-slider name="similarity-naive" min="1" max="5" required pin></crowd-slider></p>
          </li>
        <ul>
      </p>
    </crowd-form>"""

    fluency_temp = f"""
    <!-- You must include this JavaScript file -->
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

    <!-- For the full list of available Crowd HTML Elements and their input/output documentation,
          please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

    <!-- You must include crowd-form so that your task submits answers to MTurk -->
    <crowd-form answer-format="flatten-objects">

        <p>
          <strong>
            Use the sliders below to answer the questions below.
          </strong>
        </p>

      <p>Q: <strong>How likely is each sentence below written by humans?</strong> (1 = Not at all, 5 = Very much)</p>
      <p>
        <ul>
          <li>
            <p>{candidates["target"]}</p>
            <p><crowd-slider name="fluency-target" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["wst"]}</p>
            <p><crowd-slider name="fluency-wst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nst"]}</p>
            <p><crowd-slider name="fluency-nst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nps"]}</p>
            <p><crowd-slider name="fluency-nps" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["naive"]}</p>
            <p><crowd-slider name="fluency-naive" min="1" max="5" required pin></crowd-slider></p>
          </li>
        <ul>
      </p>
    </crowd-form>"""

    context_temp = f"""
    <!-- You must include this JavaScript file -->
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

    <!-- For the full list of available Crowd HTML Elements and their input/output documentation,
          please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

    <!-- You must include crowd-form so that your task submits answers to MTurk -->
    <crowd-form answer-format="flatten-objects">

        <p>
          <strong>
            Use the sliders below to answer the questions below.
          </strong>
        </p>

      <p>Q: <strong>How suitable is each sentence below as a response to the utterance by speaker A?</strong> (1 = Not at all, 5 = Very much)</p>
      <p>
          <p>Speaker A: <strong>{prev_utt}</strong></p>
        <ul>
          <li>
            <p>{candidates["target"]}</p>
            <p><crowd-slider name="context-target" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["wst"]}</p>
            <p><crowd-slider name="context-wst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nst"]}</p>
            <p><crowd-slider name="context-nst" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["nps"]}</p>
            <p><crowd-slider name="context-nps" min="1" max="5" required pin></crowd-slider></p>
          </li>
          <li>
            <p>{candidates["naive"]}</p>
            <p><crowd-slider name="context-naive" min="1" max="5" required pin></crowd-slider></p>
          </li>
        <ul>
      </p>
    </crowd-form>"""
    temps = {"emotion": emotion_temp, "similarity": similarity_temp,
             "fluency": fluency_temp, "context": context_temp}
    return temps
