import polars as pl

class InitTools:
    @staticmethod
    def init_state(file_path: str, state_dict: dict) -> str:
        """Loads a CSV into the provided state dictionary."""
        try:
            df = pl.read_csv(file_path)
            
            state_dict["df"] = df
            
            return f"Successfully loaded dataset from {file_path}. Shape: {df.shape}"
        except Exception as e:
            return f"Error loading CSV: {str(e)}"