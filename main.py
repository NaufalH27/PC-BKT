import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# ---------- Personalized BKT Function ----------
def personalized_bkt(num_students, num_skills, num_items, q_matrix, response_matrix):
    q_matrix = np.array(q_matrix)
    response_matrix = np.array(response_matrix, dtype=np.float32)

    knowledge_estimate = np.full((num_students, num_items), 0.0)
    correctness_matrix = np.where(np.isnan(response_matrix), 0, response_matrix)

    for student in range(num_students):
        for item in range(num_items):
            if not np.isnan(response_matrix[student][item]):
                skills = np.where(q_matrix[item] == 1)[0]
                if skills.size > 0:
                    knowledge_estimate[student][item] = np.mean([
                        response_matrix[student][item]
                        for skill in skills
                    ])

    P_guess = np.sum(correctness_matrix * (1 - knowledge_estimate)) / (
        np.sum(1 - knowledge_estimate) + 1e-6
    )
    P_slip = np.sum((1 - correctness_matrix) * knowledge_estimate) / (
        np.sum(knowledge_estimate) + 1e-6
    )

    initial_learning_prob = np.full((num_students, num_skills), 0.0)
    for student in range(num_students):
        for skill in range(num_skills):
            items_with_skill = [item for item in range(num_items) if q_matrix[item][skill] == 1]
            for item in items_with_skill:
                if not np.isnan(response_matrix[student][item]):
                    if response_matrix[student][item] == 1:
                        initial_learning_prob[student][skill] = 1 - P_guess
                    else:
                        initial_learning_prob[student][skill] = P_slip
                    break

    transition_prob = np.zeros(num_students)
    for student in range(num_students):
        numerator = np.sum((1 - knowledge_estimate[student, :-1]) * knowledge_estimate[student, 1:])
        denominator = np.sum(1 - knowledge_estimate[student, :-1]) + 1e-6
        transition_prob[student] = numerator / denominator

    capability_matrix = np.zeros((num_students, num_skills))
    for student in range(num_students):
        for skill in range(num_skills):
            correct_sum = 0
            count = 0
            for item in range(num_items):
                if not np.isnan(response_matrix[student][item]) and q_matrix[item][skill] == 1:
                    correct_sum += response_matrix[student][item]
                    count += 1
            capability_matrix[student][skill] = correct_sum / (count + 1e-6)

    learning_prob = np.copy(initial_learning_prob)
    for student in range(num_students):
        for skill in range(num_skills):
            for item in range(num_items):
                if q_matrix[item][skill] == 1 and not np.isnan(response_matrix[student][item]):
                    correct = response_matrix[student][item] == 1
                    prior_prob = learning_prob[student][skill]

                    if correct:
                        numerator = prior_prob * (1 - P_slip)
                        denominator = numerator + (1 - prior_prob) * P_guess
                    else:
                        numerator = prior_prob * P_slip
                        denominator = numerator + (1 - prior_prob) * (1 - P_guess)

                    posterior_prob = numerator / (denominator + 1e-6)
                    learning_prob[student][skill] = posterior_prob + (1 - posterior_prob) * transition_prob[student]

    kmeans = KMeans(n_clusters=num_skills, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(capability_matrix)

    mean_learning_by_cluster = np.zeros(num_skills)
    for cluster_id in range(num_skills):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) > 0:
            mean_learning_by_cluster[cluster_id] = np.mean(learning_prob[members])

    def predict_next_performance(student_id, skill_id):
        current_learning = learning_prob[student_id][skill_id]
        predicted_learning = current_learning * (1 - P_slip) + (1 - current_learning) * P_guess
        return predicted_learning

    return {
        "guess_prob": P_guess,
        "slip_prob": P_slip,
        "initial_learning": initial_learning_prob,
        "learning_prob": learning_prob,
        "transition_prob": transition_prob,
        "capability_matrix": capability_matrix,
        "clusters": cluster_labels,
        "mean_learning_by_cluster": mean_learning_by_cluster,
        "predict_next_performance": predict_next_performance,
    }

# ---------- Load and Process CSV ----------
df = pd.read_csv("clean_dataset.csv", nrows=20000)


# Encode IDs
user_encoder = LabelEncoder()
problem_encoder = LabelEncoder()
skill_encoder = LabelEncoder()

df["student_idx"] = user_encoder.fit_transform(df["user_id"])
df["item_idx"] = problem_encoder.fit_transform(df["problem_id"])
df["skill_idx"] = skill_encoder.fit_transform(df["skill_id"].astype(int))

num_students = df["student_idx"].nunique()
num_items = df["item_idx"].nunique()
num_skills = df["skill_idx"].nunique()

# Build response matrix
response_matrix = np.full((num_students, num_items), np.nan)
for _, row in df.iterrows():
    s = row["student_idx"]
    i = row["item_idx"]
    response_matrix[s][i] = row["correct"]

# Build Q-matrix (each item has one skill)
q_matrix = np.zeros((num_items, num_skills))
item_to_skill = df.drop_duplicates("item_idx")[["item_idx", "skill_idx"]]
for _, row in item_to_skill.iterrows():
    q_matrix[int(row["item_idx"])][int(row["skill_idx"])] = 1

# ---------- Run Personalized BKT ----------
results = personalized_bkt(
    num_students=num_students,
    num_skills=num_skills,
    num_items=num_items,
    q_matrix=q_matrix,
    response_matrix=response_matrix
)
# Create skill_id to skill_name mapping
skill_id_to_name = (
    df.drop_duplicates(subset=["skill_idx"])[["skill_idx", "skill_name"]]
    .set_index("skill_idx")["skill_name"]
    .to_dict()
)

# Mapping between internal student_idx and original user_id
idx_to_user_id = dict(enumerate(user_encoder.classes_))
user_id_to_idx = {v: k for k, v in idx_to_user_id.items()}


# Also useful: reverse mapping if needed
skill_name_to_id = {v: k for k, v in skill_id_to_name.items()}


predicted = 0
student_id = 1
skill_id = 0
while predicted == 0:
    predicted = results["predict_next_performance"](student_id, skill_id)
    student_id +=1
    skill_name = skill_id_to_name.get(skill_id, "Unknown Skill")
    actual_user_id = idx_to_user_id.get(student_id)
    print(f"Predicted performance for Student {actual_user_id} or {student_id} on Skill '{skill_name}': {predicted:.4f}")




# ---------- Cluster Summary ----------
clusters = results["clusters"]
print("\nCluster Counts:")
for cid in np.unique(clusters):
    count = np.sum(clusters == cid)
    print(f"Cluster {cid}: {count} students")

print("\nMean Learning Probability by Cluster:")
print(results["mean_learning_by_cluster"])
