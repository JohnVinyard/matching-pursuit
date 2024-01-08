import { useState } from "react";

interface VectorProps {
  vector: number[];
}

const Vector: React.FC<VectorProps> = ({ vector }) => {
  const [date, setDate] = useState<Date>(new Date());

  const update = () => {
    setDate(new Date());
  };

  return (
    <div>
      <div>Oh hai, it's {date.toString()}</div>
      <button onClick={update}>Update</button>
    </div>
  );
};

export default Vector;
