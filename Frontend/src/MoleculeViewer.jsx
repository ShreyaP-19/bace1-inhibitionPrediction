import { useEffect, useRef } from "react";
import SmilesDrawer from "smiles-drawer";

function MoleculeViewer({ smiles }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!smiles || !canvasRef.current) return;

    const drawer = new SmilesDrawer.Drawer({
      width: 400,
      height: 300
    });

    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    SmilesDrawer.parse(
      smiles,
      (tree) => {
        if (!tree) {
          console.error("Invalid SMILES → parse tree null:", smiles);
          return;
        }

        try {
          drawer.draw(tree, canvasRef.current, "light");
        } catch (e) {
          console.error("Draw error:", e);
        }
      },
      (err) => {
        console.error("SMILES parse error:", err);
      }
    );
  }, [smiles]);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 flex justify-center">
      <canvas ref={canvasRef} width={400} height={300} />
    </div>
  );
}

export default MoleculeViewer;
