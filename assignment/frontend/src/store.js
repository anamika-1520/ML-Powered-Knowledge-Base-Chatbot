import { configureStore, createSlice } from '@reduxjs/toolkit';

const interactionSlice = createSlice({
  name: 'interaction',
  initialState: { hcpName: '', topics: '', sentiment: 'Neutral' },
  reducers: {
    setForm: (state, action) => ({ ...state, ...action.payload }),
  }
});

export const { setForm } = interactionSlice.actions;
export const store = configureStore({ reducer: { interaction: interactionSlice.reducer } });