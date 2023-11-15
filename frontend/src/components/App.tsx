import {ReactQueryDevtools} from 'react-query/devtools'
import React, {useContext, useEffect} from "react";
import ExperimentTrial from "./Trials";
import {NetworkContextProvider} from "../contexts/NetworkContext";
import {QueryClient, QueryClientProvider} from 'react-query';
import {useSearchParams} from "react-router-dom";
import {v4 as uuid4} from "uuid";
import {SessionContextProvider} from "../contexts/SessionContext";


// Create a client
const queryClient = new QueryClient()



export type SearchParamsContextType = {
    prolificId: string | null;
    experimentType: string | null;
};

const SearchParamsContext = React.createContext<SearchParamsContextType | null>(null);

type AppProps = {
    experimentType: string;
  };
  


const App: React.FC<AppProps> = ({ experimentType }) => {
    const [searchParams, setSearchParams] = useSearchParams();

    useEffect(() => {
        if (!searchParams.get("PROLIFIC_PID")) {
            searchParams.set("PROLIFIC_PID", uuid4().toString());
            setSearchParams(searchParams);
        }
    }, []);


    return (
        <QueryClientProvider client={queryClient}>
            <SearchParamsContext.Provider value={
                {
                    prolificId: searchParams.get("PROLIFIC_PID"),
                    experimentType: experimentType
                }
            }>
                <SessionContextProvider prolificID={searchParams.get("PROLIFIC_PID")}>
                    <NetworkContextProvider>
                        {searchParams.get("PROLIFIC_PID") &&
                            <ExperimentTrial/>
                        }
                        <ReactQueryDevtools initialIsOpen={false}/>
                    </NetworkContextProvider>
                </SessionContextProvider>
            </SearchParamsContext.Provider>
        </QueryClientProvider>
    );
};

export default App;

export const useSearchParamsContext = () => useContext(SearchParamsContext);
