'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import Select from 'react-select'
import { Shield } from 'lucide-react'

interface Constants {
  champions: string[];
  teams: string[];
}

const constants: Constants = {
  champions: ["Aatrox", "Ahri", "Akali", "Alistar", "Amumu", /* ... other champions ... */],
  teams: ["Team SoloMid", "Cloud9", "Fnatic", "G2 Esports", /* ... other teams ... */]
}

interface TeamSelectionProps {
  side: 'blue' | 'red'
}

const TeamSelection: React.FC<TeamSelectionProps> = ({ side }) => {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null)
  const [selectedChampions, setSelectedChampions] = useState<(string | null)[]>(Array(5).fill(null))
  const [bannedChampions, setBannedChampions] = useState<(string | null)[]>(Array(5).fill(null))

  const handleChampionSelect = (index: number, value: string | null, isBan: boolean) => {
    if (isBan) {
      setBannedChampions(prev => {
        const newBans = [...prev]
        newBans[index] = value
        return newBans
      })
    } else {
      setSelectedChampions(prev => {
        const newChamps = [...prev]
        newChamps[index] = value
        return newChamps
      })
    }
  }

  return (
    <Card className={`w-full ${side === 'blue' ? 'bg-blue-50' : 'bg-red-50'}`}>
      <CardHeader>
        <CardTitle className={`text-2xl font-bold ${side === 'blue' ? 'text-blue-600' : 'text-red-600'}`}>
          {side.charAt(0).toUpperCase() + side.slice(1)} Side
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <Label htmlFor={`${side}-team`}>Team</Label>
            <Select
              id={`${side}-team`}
              options={constants.teams.map(team => ({ value: team, label: team }))}
              value={selectedTeam ? { value: selectedTeam, label: selectedTeam } : null}
              onChange={(option) => setSelectedTeam(option ? option.value : null)}
              placeholder="Select team..."
              className="mt-1"
            />
          </div>
          <div>
            <Label>Champions</Label>
            <div className="grid grid-cols-5 gap-2 mt-1">
              {selectedChampions.map((_, index) => (
                <Select
                  key={`${side}-champ-${index}`}
                  options={constants.champions.map(champ => ({ value: champ, label: champ }))}
                  value={selectedChampions[index] ? { value: selectedChampions[index], label: selectedChampions[index] } : null}
                  onChange={(option) => handleChampionSelect(index, option ? option.value : null, false)}
                  placeholder={`Champ ${index + 1}`}
                />
              ))}
            </div>
          </div>
          <div>
            <Label>Bans</Label>
            <div className="grid grid-cols-5 gap-2 mt-1">
              {bannedChampions.map((_, index) => (
                <Select
                  key={`${side}-ban-${index}`}
                  options={constants.champions.map(champ => ({ value: champ, label: champ }))}
                  value={bannedChampions[index] ? { value: bannedChampions[index], label: bannedChampions[index] } : null}
                  onChange={(option) => handleChampionSelect(index, option ? option.value : null, true)}
                  placeholder={`Ban ${index + 1}`}
                />
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default function Component() {
  const handlePredict = () => {
    // Implement prediction logic here
    console.log('Predicting...')
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-gray-800 text-white py-4">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Shield className="h-8 w-8" />
              <h1 className="text-2xl font-bold">LoL Match Predictor</h1>
            </div>
            <nav>
              <ul className="flex space-x-4">
                <li><a href="#" className="hover:text-gray-300">Home</a></li>
                <li><a href="#" className="hover:text-gray-300">About</a></li>
                <li><a href="#" className="hover:text-gray-300">Contact</a></li>
              </ul>
            </nav>
          </div>
        </div>
      </header>

      <main className="flex-grow container mx-auto px-4 py-8">
        <h2 className="text-3xl font-bold text-center mb-8">Champion Select</h2>
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <TeamSelection side="blue" />
          <TeamSelection side="red" />
        </div>
        <div className="text-center">
          <Button onClick={handlePredict} size="lg" className="bg-green-600 hover:bg-green-700 text-white">
            Predict Match Outcome
          </Button>
        </div>
      </main>

      <footer className="bg-gray-800 text-white py-4">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center">
            <p>&copy; 2023 LoL Match Predictor. All rights reserved.</p>
            <div className="flex space-x-4">
              <a href="#" className="hover:text-gray-300">Privacy Policy</a>
              <a href="#" className="hover:text-gray-300">Terms of Service</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}