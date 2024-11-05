import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Загрузка данных
df = pd.read_csv('salerefonly.csv', sep=';') # Замените на путь к вашим данным

# nom;group;mat_polok;h;n;s;p
# Предположим, что у вас есть столбцы: 'mat_polok', 'h', 'n', 's', 'p'
# Выделяем признаки и целевую переменную
X = df[['h', 'n', 'p']]
y = df['s']


# Преобразование категориальных данных (если есть)
X = pd.get_dummies(X, drop_first=True)
# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("-----------------------------------")
print("Линейная регрессия\n")

# Создаем модель
model = LinearRegression()
model.fit(X_train, y_train)
# Прогнозируем
y_pred = model.predict(X_test)
# Оценка модели
print(f'R^2 Score: {r2_score(y_test, y_pred)}')
# Коэффициенты регрессии
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

print("-----------------------------------")
print("Дерево решений\n")
# Создаем модель дерева решений
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Прогнозируем
y_pred = model.predict(X_test)

# Оценка модели
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Важность признаков
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("Feature importances:\n", feature_importances)
print("\n")
print("-----------------------------------")
print("Градиентный бустинг\n")

# Создаем модель градиентного бустинга
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Прогнозируем
y_pred = model.predict(X_test)

# Оценка модели
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Важность признаков
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("Feature importances:\n", feature_importances)
print("\n")
print("-----------------------------------")
print("Случайный лес\n")
# Создаем модель случайного леса
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Прогнозируем
y_pred = model.predict(X_test)

# Оценка модели
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Важность признаков
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("Feature importances:\n", feature_importances)
print("\n")
print("-----------------------------------")
print("SVM\n")

# Создаем модель SVM для регрессии
model = SVR(kernel='linear')  # Можно попробовать другие ядра, например, 'poly', 'rbf'
model.fit(X_train, y_train)

# Прогнозируем
y_pred = model.predict(X_test)

# Оценка модели
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Коэффициенты регрессии
coefficients = model.coef_
print(f'Coefficients: {coefficients}')

print("\n")
print("-----------------------------------")
print("Нейронные сети\n")

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создаем модель нейронной сети
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer='adam', loss='mse')

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Прогнозируем
y_pred = model.predict(X_test)

# Оценка модели
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')