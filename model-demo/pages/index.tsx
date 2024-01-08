import ModelDemoView from "../components/ModelDemoView";

interface HomeProps {}

const Home: React.FC<HomeProps> = ({}) => {
  return <ModelDemoView nReconstructions={3} />;
};

export default Home;
