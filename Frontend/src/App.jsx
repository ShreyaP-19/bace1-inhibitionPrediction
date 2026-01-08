import React, { useState } from "react";
import {
  Atom,
  Search,
  Activity,
  Trash2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Beaker
} from "lucide-react";

export default function App() {
  const [smiles, setSmiles] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const handleClear = () => {
    setSmiles("");
    setResult(null);
    setError("");
  };

  const handlePredict = () => {
    setError("");
    setResult(null);

    if (!smiles.trim()) {
      setError("Please enter a valid SMILES string.");
      return;
    }

    setIsAnalyzing(true);

    // Simulate API call
    setTimeout(() => {
      setIsAnalyzing(false);

      // Mock random prediction logic
      const isInhibitor = Math.random() > 0.5;
      const predictedIC50 = (Math.random() * (9.0 - 4.0) + 4.0).toFixed(2); // Random pIC50 between 4 and 9
      const confidenceScore = Math.random().toFixed(2);

      setResult({
        isInhibitor,
        predictedIC50,
        confidenceScore
      });
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4 md:p-6 font-sans text-slate-900">

      {/* Main Container */}
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-200">

        {/* Header */}
        <header className="bg-slate-900 text-white p-8 text-center relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500 opacity-5 rounded-full blur-3xl translate-x-1/2 -translate-y-1/2"></div>
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-blue-600 opacity-10 rounded-full blur-2xl -translate-x-1/2 translate-y-1/2"></div>

          <div className="relative z-10 flex flex-col items-center">
            <div className="bg-white/10 p-4 rounded-2xl mb-4 backdrop-blur-sm border border-white/10 shadow-lg">
              <Atom className="w-10 h-10 text-cyan-400" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold mb-2 tracking-tight">
              BACE-1 Inhibitor Prediction
            </h1>
            <p className="text-slate-300 max-w-lg mx-auto text-sm md:text-base leading-relaxed">
              Advanced deep learning model for predicting Beta-secretase 1 inhibition potency from molecular structure.
            </p>
          </div>
        </header>

        <main className="p-6 md:p-8 space-y-8">

          {/* Input Section */}
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <label htmlFor="smiles-input" className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
                Input Molecule (SMILES)
              </label>
              <button
                onClick={handleClear}
                className="text-slate-400 hover:text-red-500 text-sm flex items-center gap-1 transition-colors"
                disabled={!smiles && !result}
              >
                <Trash2 className="w-4 h-4" />
                Clear
              </button>
            </div>

            <div className="relative group">
              <textarea
                id="smiles-input"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
                className={`w-full h-32 pl-4 pr-4 py-3 bg-slate-50 border-2 ${error ? 'border-red-300 focus:border-red-400 focus:ring-red-100' : 'border-slate-200 focus:border-blue-400 focus:ring-blue-50'} rounded-xl focus:outline-none focus:ring-4 transition-all duration-200 text-slate-700 placeholder:text-slate-400 font-mono text-sm resize-none shadow-sm`}
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 text-red-600 text-sm animate-pulse">
                <AlertCircle className="w-4 h-4" />
                <span>{error}</span>
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={isAnalyzing || !smiles.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-xl shadow-lg shadow-blue-200 transition-all duration-200 transform active:scale-[0.99] flex justify-center items-center gap-3"
            >
              {isAnalyzing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Analyzing Structure...</span>
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  <span>Run Prediction</span>
                </>
              )}
            </button>
          </div>

          {/* Results Section */}
          {result && (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 space-y-6">
              <div className="border-t border-slate-100 pt-8">
                <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  Prediction Results
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                  {/* Classification Card */}
                  <div className={`p-6 rounded-2xl border-2 ${result.isInhibitor ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'} flex flex-col items-center justify-center text-center shadow-sm`}>
                    <div className={`p-4 rounded-full mb-4 ${result.isInhibitor ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>
                      {result.isInhibitor ? <CheckCircle2 className="w-10 h-10" /> : <XCircle className="w-10 h-10" />}
                    </div>
                    <h3 className="text-slate-500 text-xs font-bold uppercase tracking-wider mb-1">Classification</h3>
                    <p className={`text-2xl font-bold ${result.isInhibitor ? 'text-green-800' : 'text-red-800'}`}>
                      {result.isInhibitor ? "BACE-1 Inhibitor" : "Non-Inhibitor"}
                    </p>
                    <p className="text-slate-500 text-sm mt-2">
                      Confidence: <span className="font-semibold text-slate-700">{Math.round(result.confidenceScore * 100)}%</span>
                    </p>
                  </div>

                  {/* Regression Card */}
                  <div className="p-6 rounded-2xl border border-slate-200 bg-white shadow-sm flex flex-col justify-center">
                    <div className="flex justify-between items-end mb-4">
                      <div className="flex items-center gap-2">
                        <Beaker className="w-5 h-5 text-slate-400" />
                        <h3 className="text-slate-500 text-xs font-bold uppercase tracking-wider">Potency (pIC50)</h3>
                      </div>
                      <span className="text-3xl font-bold text-slate-800">{result.predictedIC50}</span>
                    </div>

                    {/* Progress Bar */}
                    <div className="h-4 w-full bg-slate-100 rounded-full overflow-hidden relative">
                      <div
                        className="h-full bg-gradient-to-r from-blue-400 to-indigo-600 transition-all duration-1000 ease-out rounded-full"
                        style={{ width: `${(result.predictedIC50 / 10) * 100}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-slate-400 mt-2 font-mono">
                      <span>0</span>
                      <span>5</span>
                      <span>10+</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-4 leading-relaxed">
                      Predicted negative log of half maximal inhibitory concentration (-log IC50). Higher values indicate greater potency.
                    </p>
                  </div>
                </div>
              </div>

              {/* Molecule Visualization Placeholder */}
              <div className="bg-slate-50 border border-slate-200 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center min-h-[200px] text-center">
                <div id="mol-structure" className="w-full h-full flex items-center justify-center">
                  <div className="text-slate-400 flex flex-col items-center gap-3">
                    <Atom className="w-12 h-12 opacity-20" />
                    <span className="text-sm font-medium">Structure Visualization Placeholder</span>
                    <span className="text-xs max-w-xs opacity-70">
                      (Integration with RDKit.js or SmilesDrawer would be rendered here)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="fixed bottom-4 text-center w-full pointer-events-none">
        <p className="text-slate-400 text-xs font-medium bg-white/50 backdrop-blur-md py-1 px-3 rounded-full inline-block shadow-sm">
          BACE-1 Research Project • 2026
        </p>
      </footer>
    </div>
  );
}