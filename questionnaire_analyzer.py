#!/usr/bin/env python3
"""
アンケート画像解析ツール
BBox座標を使用して画像を切り出し、Gemini APIで構造化解析を行います。
"""

import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import json
import io
import os
import re
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# BBox座標（提供された座標）
BBOX_COORDINATES = {
    "A": [47, 156, 880, 499],  # 仕事について
    "B": [47, 504, 880, 902],  # 最近1か月間の状態
    "C": [47, 906, 881, 1083], # 周りの方々について
    "D": [45, 1083, 320, 1200] # 満足度について
}

def crop_image_by_bbox(image, bbox):
    """
    画像をBBox座標で切り出す
    
    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2] 座標リスト
        
    Returns:
        PIL Image: 切り出された画像
    """
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))

def extract_json_from_response(response_text):
    """
    レスポンステキストからJSONを抽出する
    
    Args:
        response_text: Gemini APIからのレスポンス
        
    Returns:
        dict: 抽出されたJSON、またはエラー情報
    """
    try:
        # まず、そのままJSONとして解析を試行
        return json.loads(response_text)
    except json.JSONDecodeError:
        # JSONが見つからない場合、コードブロック内を探す
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 波括弧で囲まれた部分を探す
        brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {
            "error": "JSON解析エラー",
            "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
        }

def analyze_section_with_gemini(image, section_name, api_key, model_name):
    """
    Gemini APIを使用して画像セクションを解析
    
    Args:
        image: PIL Image
        section_name: セクション名 (A, B, C, D)
        api_key: Gemini APIキー
        model_name: 使用するGeminiモデル名
        
    Returns:
        dict: 解析結果
    """
    if not api_key:
        return {"error": "Gemini APIキーが設定されていません"}
    
    try:
        # APIキーを設定
        genai.configure(api_key=api_key)
        
        # 画像をバイト形式に変換
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Geminiモデルを初期化
        model = genai.GenerativeModel(model_name)
        
        # セクション別のプロンプト
        prompts = {
            "A": """
            この画像はアンケートの「あなたの仕事について」のセクションです。
            各質問項目について、どの選択肢（そうだ、まあそうだ、ややちがう、ちがう）にチェックが入っているかを読み取ってください。

            重要: 必ず以下のJSON形式で回答してください。他の説明は不要です。

            {
                "section": "A",
                "title": "あなたの仕事について",
                "questions": [
                    {
                        "number": 1,
                        "question": "非常にたくさんの仕事をしなければならない",
                        "answer": "そうだ"
                    }
                ]
            }
            """,
            "B": """
            この画像はアンケートの「最近1か月間のあなたの状態について」のセクションです。
            各質問項目について、どの選択肢（ほとんどいつもあった、しばしばあった、ときどきあった、ほとんどなかった）にチェックが入っているかを読み取ってください。

            重要: 必ず以下のJSON形式で回答してください。他の説明は不要です。

            {
                "section": "B",
                "title": "最近1か月間のあなたの状態について",
                "questions": [
                    {
                        "number": 1,
                        "question": "活気がわいてくる",
                        "answer": "ほとんどなかった"
                    }
                ]
            }
            """,
            "C": """
            この画像はアンケートの「あなたの周りの方々について」のセクションです。
            各質問項目について、どの選択肢（非常に、かなり、多少、ない、全くない）にチェックが入っているかを読み取ってください。

            重要: 必ず以下のJSON形式で回答してください。他の説明は不要です。

            {
                "section": "C",
                "title": "あなたの周りの方々について",
                "questions": [
                    {
                        "number": 1,
                        "question": "上司",
                        "answer": "非常に"
                    }
                ]
            }
            """,
            "D": """
            この画像はアンケートの「満足度について」のセクションです。
            各質問項目について、どの選択肢（満足、まあ満足、やや不満足、不満足）にチェックが入っているかを読み取ってください。

            重要: 必ず以下のJSON形式で回答してください。他の説明は不要です。

            {
                "section": "D",
                "title": "満足度について",
                "questions": [
                    {
                        "number": 1,
                        "question": "仕事に満足だ",
                        "answer": "満足"
                    }
                ]
            }
            """
        }
        
        # Gemini APIで解析
        response = model.generate_content([
            prompts[section_name],
            {"mime_type": "image/png", "data": img_byte_arr.getvalue()}
        ])
        
        # レスポンスからJSONを抽出
        result = extract_json_from_response(response.text)
        return result
            
    except Exception as e:
        return {"error": f"Gemini API エラー: {str(e)}"}

def main():
    st.title("アンケート画像解析ツール")
    st.write("アンケート画像をアップロードして、各セクションの回答を構造化解析します。")
    
    # サイドバー設定
    st.sidebar.header("設定")
    
    # APIキーの設定
    api_key = st.sidebar.text_input("Gemini APIキー", type="password", help="Google AI Studioから取得したAPIキーを入力してください")
    
    # モデル選択
    model_options = {
        "Gemini 1.5 Flash (高速)": "gemini-1.5-flash",
        "Gemini 1.5 Pro (高精度)": "gemini-1.5-pro",
        "Gemini 2.0 Flash (最新)": "gemini-2.0-flash-exp",
        "Gemini 2.5 Pro (最高精度)": "gemini-2.5-pro"
    }
    
    selected_model = st.sidebar.selectbox(
        "Geminiモデルを選択",
        list(model_options.keys()),
        index=3,  # Gemini 2.5 Proをデフォルトに設定
        help="最高精度が必要な場合はGemini 2.5 Proを選択してください"
    )
    
    model_name = model_options[selected_model]
    
    if api_key:
        st.sidebar.success("APIキーが設定されました")
        st.sidebar.info(f"選択されたモデル: {selected_model}")
    else:
        st.sidebar.warning("Gemini APIキーを設定してください")
    
    # 画像アップロード
    uploaded_file = st.file_uploader("アンケート画像をアップロード", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 画像を読み込み
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされた画像", use_container_width=True)
        
        # BBox座標の表示
        st.subheader("BBox座標")
        st.json(BBOX_COORDINATES)
        
        # 解析ボタン
        if st.button("画像を解析"):
            if not api_key:
                st.error("Gemini APIキーを設定してください")
                return
            
            with st.spinner("画像を解析中..."):
                results = {}
                
                # 各セクションを解析
                for section, bbox in BBOX_COORDINATES.items():
                    st.write(f"セクション{section}を解析中...")
                    
                    # 画像を切り出し
                    cropped_image = crop_image_by_bbox(image, bbox)
                    
                    # 切り出し画像を表示
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cropped_image, caption=f"セクション{section}の切り出し", use_container_width=True)
                    
                    # Gemini APIで解析
                    with col2:
                        result = analyze_section_with_gemini(cropped_image, section, api_key, model_name)
                        results[section] = result
                        
                        if "error" in result:
                            st.error(f"セクション{section}の解析エラー: {result['error']}")
                            if "raw_response" in result:
                                with st.expander("生レスポンスを表示"):
                                    st.text(result["raw_response"])
                        else:
                            st.success(f"セクション{section}の解析完了")
                            st.json(result)
                
                # 全体結果の表示
                st.subheader("解析結果")
                st.json(results)
                
                # 結果をダウンロード
                json_str = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="解析結果をダウンロード",
                    data=json_str,
                    file_name="questionnaire_analysis.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main() 