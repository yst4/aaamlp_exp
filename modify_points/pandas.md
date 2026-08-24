この本で頻出する Pandas の地雷ポイントと、その最新の書き換えパターンです。

df.append() の全滅

旧: df = df.append(new_row, ignore_index=True)

新: df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

inplace=True の非推奨化・非推奨警告

旧: df.drop(columns=['id'], inplace=True)

新: df = df.drop(columns=['id'])

iloc / loc での型厳格化（Series.values 参照）

旧: df.iloc[0, :].values.reshape(...)

新: df.iloc[0].to_numpy().reshape(...)

groupby 後の一部挙動や as_index の変化

型推論の厳格化により、文字列と数値が混在したカラムの処理で即エラー化

特に交差検証（3章）で Fold 用のカラムを追加して df.to_csv(...) で書き出す処理や、カテゴリ変数のエンコーディング（4章）あたりでバシバシ引っかかるはずです。

「警告が出た」「エラーで止まった」となったら、該当コードをペッと貼ってもらえれば一瞬で現代風の書き方に直します！
