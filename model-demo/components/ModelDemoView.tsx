import { useEffect, useState } from "react";
import { Suggestion } from "../models/Suggestion";

interface ModelDemoViewProps {
  nReconstructions: number;
}

const ModelDemoView: React.FC<ModelDemoViewProps> = ({}) => {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);

  useEffect(() => {
    fetch("/suggestions")
      .then((resp) => resp.json())
      .then((suggestions) => {
        setSuggestions(suggestions);
        console.log(suggestions);
      });
  }, []);
  
  return (
    <ul>
      {suggestions.map((suggestion) => (
        <li>{suggestion.encoding}</li>
      ))}
    </ul>
  );
};

export default ModelDemoView;
