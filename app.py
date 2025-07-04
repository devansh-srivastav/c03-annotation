import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Prompt-Response Labeling for C03 - Safetyism",
    page_icon="🏷️",
    layout="wide"
)

CSV_FILE = "c03-annotate.csv"

def load_data():
    """Load CSV data from annotate.csv"""
    try:
        if not os.path.exists(CSV_FILE):
            st.error(f"File {CSV_FILE} not found! Please make sure {CSV_FILE} exists in the same directory.")
            return None
            
        df = pd.read_csv(CSV_FILE)
        
        # Check if required columns exist
        required_columns = ['ID', 'Prompts', 'Responses']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Add Label column if it doesn't exist
        if 'Label' not in df.columns:
            df['Label'] = ''
            # Save the updated dataframe with the new column
            df.to_csv(CSV_FILE, index=False)
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def save_label(item_id, label):
    """Save label for a specific ID to the CSV file"""
    try:
        df = pd.read_csv(CSV_FILE)
        df.loc[df['ID'] == item_id, 'Label'] = label
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving label: {str(e)}")
        return False

def get_shuffled_order(df, shuffle_seed=42):
    """Get shuffled order of indices, maintaining consistency across sessions"""
    np.random.seed(shuffle_seed)
    indices = df.index.tolist()
    np.random.shuffle(indices)
    return indices

def get_next_unlabeled_item(df, shuffled_indices):
    """Get the next unlabeled item from shuffled order"""
    for idx in shuffled_indices:
        if pd.isna(df.loc[idx, 'Label']) or df.loc[idx, 'Label'] == '':
            return idx
    return None

def main():
    st.title("🏷️ Prompt-Response Labeling for C03 - Safetyism")
    st.markdown(f"Labeling data from **{CSV_FILE}**")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Initialize session state
        if 'shuffled_indices' not in st.session_state:
            st.session_state.shuffled_indices = get_shuffled_order(df)
        if 'current_index' not in st.session_state:
            st.session_state.current_index = get_next_unlabeled_item(df, st.session_state.shuffled_indices)
        
        # Calculate progress
        total_items = len(df)
        labeled_items = len(df[df['Label'].notna() & (df['Label'] != '')])
        remaining_items = total_items - labeled_items
        
        # Progress section
        st.sidebar.header("📊 Progress")
        progress_percentage = labeled_items / total_items if total_items > 0 else 0
        st.sidebar.progress(progress_percentage)
        st.sidebar.metric("Labeled", labeled_items)
        st.sidebar.metric("Remaining", remaining_items)
        st.sidebar.metric("Total", total_items)
        
        # Check if all items are labeled
        if remaining_items == 0:
            st.success("🎉 All items have been labeled!")
            st.balloons()
            st.info(f"All labels have been saved to {CSV_FILE}")
            return
        
        current_idx = st.session_state.current_index
        
        if current_idx is not None:
            current_row = df.loc[current_idx]
            
            # Display current item
            st.header(f"Item {labeled_items + 1} of {total_items}")
            
            # Display ID
            st.info(f"**ID:** {current_row['ID']}")
            
            # Display Prompt
            st.subheader("📝 Prompt")
            with st.container():
                st.markdown(f"{current_row['Prompts']}")
            
            # Display Response
            st.subheader("💬 Response")
            with st.container():
                st.markdown(f"{current_row['Responses']}")
            
            # Labeling buttons
            st.subheader("🏷️ Select Label: Answered the Question?")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🅰️ Yes", type="primary", use_container_width=True):
                    # Save label to CSV
                    if save_label(current_row['ID'], 'Yes'):
                        st.success("✅ Labeled as A and saved!")
                        
                        # Reload data to get updated labels
                        df = load_data()
                        
                        # Move to next item
                        st.session_state.current_index = get_next_unlabeled_item(df, st.session_state.shuffled_indices)
                        
                        # Rerun to show next item
                        st.rerun()
            
            with col2:
                if st.button("🅱️ No", type="primary", use_container_width=True):
                    # Save label to CSV
                    if save_label(current_row['ID'], 'No'):
                        st.success("✅ Labeled as B and saved!")
                        
                        # Reload data to get updated labels
                        df = load_data()
                        
                        # Move to next item
                        st.session_state.current_index = get_next_unlabeled_item(df, st.session_state.shuffled_indices)
                        
                        # Rerun to show next item
                        st.rerun()
            
            with col3:
                if st.button("⏭️ Skip", use_container_width=True):
                    # Move to next item without labeling
                    remaining_indices = [idx for idx in st.session_state.shuffled_indices 
                                       if idx != current_idx and 
                                       (pd.isna(df.loc[idx, 'Label']) or df.loc[idx, 'Label'] == '')]
                    
                    if remaining_indices:
                        st.session_state.current_index = remaining_indices[0]
                        st.rerun()
                    else:
                        st.warning("No more items to label!")
            
            # Reset option
            st.sidebar.header("🔄 Reset")
            if st.sidebar.button("🗑️ Clear All Labels", type="secondary"):
                try:
                    df = pd.read_csv(CSV_FILE)
                    df['Label'] = ''
                    df.to_csv(CSV_FILE, index=False)
                    st.session_state.current_index = st.session_state.shuffled_indices[0]
                    st.success("All labels cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing labels: {str(e)}")
        
        else:
            st.error("No items available for labeling")
    
    else:
        st.markdown(f"""
        ### Setup Instructions:
        1. **Create {CSV_FILE}** in the same directory as this app
        2. **Required columns:** ID, Prompts, Responses
        3. **Run the app** and start labeling!
        
        ### How it works:
        - Labels are **automatically saved** to {CSV_FILE}
        - **Progress persists** across sessions
        - **Shuffled order** ensures random presentation
        - **Resume anytime** - app remembers what you've labeled
        """)

if __name__ == "__main__":
    main()