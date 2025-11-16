"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""



def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings('ignore')

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')


    def preprocess_data(train_df, test_df):
        train_processed = train_df.copy()
        test_processed = test_df.copy()

        categorical_cols = ['zip_code', 'channel', 'history_segment']
        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            train_processed[col] = le.fit_transform(train_processed[col].astype(str))
            test_processed[col] = test_processed[col].astype(str)
            mask = test_processed[col].isin(le.classes_)
            if not mask.all():
                test_processed.loc[~mask, col] = le.classes_[0]
            test_processed[col] = le.transform(test_processed[col])
            encoders[col] = le

        return train_processed, test_processed, encoders

    train_processed, test_processed, encoders = preprocess_data(train, test)

    feature_cols = ['recency', 'history', 'mens', 'womens', 'newbie',
                    'zip_code', 'channel', 'history_segment']

    X_train = train_processed[feature_cols]
    y_visit = train_processed['visit']
    actions = train_processed['segment']


    def create_contextual_features(df):
        df_fe = df.copy()

        df_fe['total_interest'] = df_fe['mens'] + df_fe['womens']
        df_fe['interest_ratio'] = (df_fe['mens'] + 1) / (df_fe['womens'] + 1)
        df_fe['recency_history_interaction'] = df_fe['recency'] * df_fe['history']
        df_fe['activity_score'] = df_fe['history'] / (df_fe['recency'] + 1)

        df_fe['high_value'] = (df_fe['history'] > df_fe['history'].quantile(0.7)).astype(int)
        df_fe['recent_active'] = (df_fe['recency'] < df_fe['recency'].quantile(0.3)).astype(int)
        df_fe['multichannel_premium'] = ((df_fe['channel'] == 1) &
                                        (df_fe['history'] > df_fe['history'].median())).astype(int)

        return df_fe

    X_train_fe = create_contextual_features(X_train)
    X_test_fe = create_contextual_features(test_processed[feature_cols])

    X_train_fe = X_train_fe.reset_index(drop=True)
    X_test_fe = X_test_fe.reset_index(drop=True)
    actions = actions.reset_index(drop=True)
    y_visit = y_visit.reset_index(drop=True)


    class ContextualBandit:
        def __init__(self, n_actions=3, temperature=1.0, epsilon=0.1):
            self.n_actions = n_actions
            self.temperature = temperature
            self.epsilon = epsilon
            self.action_mapping = {'Mens E-Mail': 0, 'Womens E-Mail': 1, 'No E-Mail': 2}
            self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}

        def fit_reward_model(self, X, actions, rewards):
            self.reward_models = {}
            for action_name, action_idx in self.action_mapping.items():
                mask = (actions == action_name)
                if mask.sum() > 10:
                    model = GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=6,
                        min_samples_leaf=20,
                        learning_rate=0.1,
                        random_state=42
                    )
                    model.fit(X[mask], rewards[mask])
                    self.reward_models[action_idx] = model

        def predict_action_values(self, X):
            n_samples = len(X)
            action_values = np.zeros((n_samples, self.n_actions))
            for action_idx in range(self.n_actions):
                if action_idx in self.reward_models:
                    action_values[:, action_idx] = self.reward_models[action_idx].predict_proba(X)[:, 1]
                else:
                    action_values[:, action_idx] = 0.1
            return action_values

        def value_to_policy(self, action_values, temperature=None):
            if temperature is None:
                temperature = self.temperature
            exp_values = np.exp(action_values / temperature - np.max(action_values / temperature, axis=1, keepdims=True))
            policy = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            policy = self.apply_epsilon_greedy(policy)
            return self.apply_safety_constraints(policy)

        def apply_epsilon_greedy(self, policy):
            return (1 - self.epsilon) * policy + self.epsilon / self.n_actions

        def apply_safety_constraints(self, policy):
            policy = np.clip(policy, 0.05, 0.95)
            policy = policy / np.sum(policy, axis=1, keepdims=True)
            return policy

        def fit_ips_policy(self, X, actions, rewards, behavior_policy=None, num_iterations=200, learning_rate=0.05):
            n_samples, n_features = X.shape
            self.weights = np.random.randn(n_features, self.n_actions) * 0.01

            action_indices = np.array([self.action_mapping[a] for a in actions])

            if behavior_policy is None:
                behavior_policy = np.full((n_samples, self.n_actions), 1.0 / self.n_actions)

            for iteration in range(num_iterations):
                logits = X @ self.weights
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                policy_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                dr_rewards = np.zeros_like(policy_probs)
                dr_rewards[np.arange(n_samples), action_indices] = rewards / behavior_policy[np.arange(n_samples), action_indices]
                dr_rewards -= np.mean(dr_rewards, axis=0)

                gradient = X.T @ (policy_probs * np.sum(dr_rewards, axis=1)[:, None] - dr_rewards)
                self.weights += learning_rate * gradient / n_samples

        def predict_ips_policy(self, X, epsilon=None):
            logits = X @ self.weights
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            policy = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            if epsilon is not None:
                policy = (1 - epsilon) * policy + epsilon / self.n_actions
            return self.apply_safety_constraints(policy)

    def apply_business_rules(policy_probs, X):
        adjusted_probs = policy_probs.copy()

        mens_interest = X['mens'] == 1
        womens_interest = X['womens'] == 1
        new_customers = X['newbie'] == 1
        high_value = X['history'] > X['history'].median()

        adjusted_probs[mens_interest, 0] *= 1.3
        adjusted_probs[mens_interest, 1] *= 0.8

        adjusted_probs[womens_interest, 0] *= 0.8
        adjusted_probs[womens_interest, 1] *= 1.3

        adjusted_probs[new_customers, 2] *= 0.7
        adjusted_probs[high_value, 2] *= 0.8

        row_sums = adjusted_probs.sum(axis=1)
        adjusted_probs = adjusted_probs / row_sums[:, np.newaxis]

        return adjusted_probs

    bandit = ContextualBandit(temperature=0.5, epsilon=0.1)

    bandit.fit_reward_model(X_train_fe, actions, y_visit)
    dm_policy_test = bandit.value_to_policy(bandit.predict_action_values(X_test_fe))

    bandit.fit_ips_policy(X_train_fe.values, actions, y_visit.values)
    ips_policy = bandit.predict_ips_policy(X_test_fe.values, epsilon=0.05)

    final_policy = 0.7 * ips_policy + 0.3 * dm_policy_test
    final_policy = apply_business_rules(final_policy, X_test_fe)
    final_policy = bandit.apply_safety_constraints(final_policy)

    submission = pd.DataFrame({
        'id': test['id'],
        'p_mens_email': final_policy[:, 0],
        'p_womens_email': final_policy[:, 1],
        'p_no_email': final_policy[:, 2]
    })

    prob_cols = ['p_mens_email', 'p_womens_email', 'p_no_email']
    submission[prob_cols] = submission[prob_cols].div(submission[prob_cols].sum(axis=1), axis=0)
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
